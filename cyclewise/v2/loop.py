"""CycleWise v2 headless loop: sequential checkpoints at 50/100/150 (config/prereg_v2.yaml).

Per batch (b1 train, b2 validation, b3 test):
  Evidence (ΔQ base signal) -> safety scan ->
  checkpoint 50:  Planner picks how many to continue -> HUMAN APPROVAL -> commit -> pay 50->100
  checkpoint 100: same                                                    -> pay 100->150
  checkpoint 150: Planner picks the final ranking -> APPROVAL -> commit -> keep k_final to EOL
  Critic: trigger on revealed cells; may refit the 150 rule (never on b3).

Run:  python -m cyclewise.v2.loop [--run-id ID] [--batches b1 b2 b3]
"""

from __future__ import annotations

import argparse
import json
import uuid

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut

from cyclewise import prereg
from cyclewise.agents import approval, llm
from cyclewise.config import load_config, path
from cyclewise.record import research_log
from cyclewise.record.dblock import connect_ro
from cyclewise.v2 import agents, features, visible
from cyclewise.v2.allocation import Budget, cut_to_budget, options_at_50, options_at_100, options_at_150
from cyclewise.v2.data import SCREEN

DQ_ONLY = {"kind": "zscore_sum", "fit_on": "none", "weights": {"dq_logvar": -1.0}}
ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


class RunStopped(Exception):
    pass


def roles() -> dict[str, str]:
    with connect_ro(path(SCREEN)) as con:
        return dict(con.execute("SELECT batch, role FROM roles ORDER BY batch").fetchall())


def safety(run_id: str, batch: str, cells: list[str]) -> list[dict]:
    cfg = load_config()["safety"]
    df = visible.query_cycles(run_id, cells, ["tmax", "ir", "glitch"], 50)
    df = df[(df["cycle"] >= 2) & ~df["glitch"].astype(bool)]
    flags, tm = [], {}
    for cid, g in df.groupby("cell_id"):
        t, ir = g["tmax"].replace(0, np.nan), g["ir"].replace(0, np.nan).dropna()
        if t.notna().sum() < 5:
            flags.append({"cell_id": cid, "flag": "sensor_missing", "detail": "temperature trace missing"})
            continue
        tm[cid] = t.mean()
        jump = float(ir.diff().abs().max()) if len(ir) > 1 else 0.0
        if jump > cfg["ir_jump_threshold_ohm"]:
            flags.append({"cell_id": cid, "flag": "ir_jump", "detail": f"max IR step {jump * 1000:.2f} mOhm"})
    if len(tm) > 1:
        v = pd.Series(tm)
        z = (v - v.mean()) / (v.std(ddof=1) or 1.0)
        flags += [{"cell_id": c, "flag": "hot_cell", "detail": f"early Tmax z={zz:.2f}"}
                  for c, zz in z.items() if zz > cfg["temp_z_threshold"]]
    flags.sort(key=lambda f: (f["cell_id"], f["flag"]))
    research_log.append(run_id, "critic_safety", "safety_flags", flags, batch=batch)
    return flags


def render(batch: str, ckpt: int, plan_id: str, chosen: dict, options: list[dict], reason: str,
           selected: list[str], stopped: list[str], flags: list[dict], budget_line: str) -> str:
    lines = [f"=== APPROVAL REQUEST  {plan_id}  batch {batch}  checkpoint {ckpt} ===",
             f"Chosen [{chosen['option_id']}] {chosen['description']}", f"Reason: {reason}", budget_line]
    for o in options:
        lines.append(f" {'*' if o is chosen else ' '} [{o['option_id']}] {o['description']}")
    verb = "Keep to end of life" if ckpt == 150 else f"Continue to cycle {ckpt + 50}"
    lines.append(f"{verb} ({len(selected)}): {', '.join(selected)}")
    lines.append(f"Stop now ({len(stopped)}): {', '.join(stopped) if stopped else '-'}")
    fl = [f for f in flags if f["cell_id"] in selected]
    lines.append("Safety flags among selected: " + ("; ".join(f"{f['cell_id']} {f['flag']}: {f['detail']}" for f in fl)
                                                     if fl else "none"))
    return "\n".join(lines)


def checkpoint(run_id: str, batch: str, ckpt: int, options: list[dict], rankings: dict[str, list[str]],
               flags: list[dict], remaining: int, context: dict) -> tuple[list[str], str]:
    """Planner choice -> approval (one re-proposal on rejection) -> commit. Returns (selected, plan_id)."""
    note = ""
    for attempt in (1, 2):
        ctx = dict(context, **({"human_rejection_note": note} if note else {}))
        chosen, reason, fb = agents.planner(run_id, batch, ckpt, options, ctx)
        ranked = rankings[chosen["ranking"]]
        if ckpt == 150:
            selected, was_cut = ranked[: chosen["m"]], False
        else:
            selected, was_cut = cut_to_budget(ranked, chosen["m"], remaining)
        stopped = [c for c in ranked if c not in selected]
        plan_id = f"{batch}-c{ckpt}-{uuid.uuid4().hex[:6]}"
        plan = {"plan_id": plan_id, "batch": batch, "checkpoint": ckpt, "options": options,
                "planned_m150": chosen.get("m150_planned"),
                "chosen": chosen["option_id"], "reason": reason, "selected": selected, "stopped": stopped,
                "cut_at_budget": was_cut, "remaining_budget": remaining, "flags": flags}
        research_log.append(run_id, "planner", "checkpoint_plan", plan, batch=batch, inputs=context,
                            model=llm.model_name() if not fb else "offline-deterministic",
                            prompt_version=agents.prompt_version(), fallback_used=fb)
        text = render(batch, ckpt, plan_id, chosen, options, reason, selected, stopped, flags,
                      f"Remaining extension budget: {remaining} channel-cycles")
        flagged_sel = [f["cell_id"] for f in flags if f["cell_id"] in selected]
        dec = approval.decide(run_id, plan_id, batch, text, json.dumps(plan, default=str), flagged_sel)
        if dec.decision == "approved":
            if dec.excluded_flagged:
                selected = [c for c in selected if c not in dec.excluded_flagged]
            key = "keep" if ckpt == 150 else "extend"
            research_log.append(run_id, "runner", "checkpoint_committed",
                                {"plan_id": plan_id, "checkpoint": ckpt, key: sorted(selected),
                                 "keep" if key == "extend" else "extend": [],
                                 "dropped_flagged": dec.excluded_flagged}, batch=batch)
            return selected, plan_id
        note = dec.note or "rejected"
        print(f"[human] rejected (attempt {attempt}); planner re-proposes once with the note")
    research_log.append(run_id, "orchestrator", "run_stopped",
                        {"reason": f"checkpoint {ckpt} rejected twice", "batch": batch}, batch=batch)
    raise RunStopped(f"batch {batch} checkpoint {ckpt}: rejected twice")


def fit_ridge(pool: pd.DataFrame, feats: list[str], fit_on: str) -> dict:
    X = pool[feats].astype(float)
    X = X.fillna(X.median())
    y = np.log10(pool["cycle_life"].astype(float)).values
    mean, std = X.mean(), X.std(ddof=0).replace(0, 1.0)
    Z = ((X - mean) / std).values
    best = None
    for a in ALPHAS:
        err = [(Ridge(alpha=a).fit(Z[tr], y[tr]).predict(Z[te])[0] - y[te][0]) ** 2
               for tr, te in LeaveOneOut().split(Z)]
        if best is None or np.mean(err) < best[1]:
            best = (a, float(np.mean(err)))
    m = Ridge(alpha=best[0]).fit(Z, y)
    return {"kind": "ridge", "features": feats, "target": "log10_cycle_life", "checkpoint": 150,
            "mean": {f: float(mean[f]) for f in feats}, "std": {f: float(std[f]) for f in feats},
            "coef": {f: float(c) for f, c in zip(feats, m.coef_)}, "intercept": float(m.intercept_),
            "alpha": best[0], "loo_rmse_log10": round(best[1] ** 0.5, 4), "fit_on": fit_on, "n_train": len(pool)}


def run_batch(run_id: str, batch: str, role: str, state: dict) -> None:
    paid = visible.paid(run_id, batch)
    cells = paid["cell_id"].tolist()
    budget = Budget.for_batch(len(cells))
    research_log.append(run_id, "orchestrator", "batch_start",
                        {"role": role, "n": budget.n, "k_final": budget.k_final,
                         "extension_budget": budget.extension}, batch=batch)
    print(f"\n======== v2 BATCH {batch} (role={role}) N={budget.n} k_final={budget.k_final} "
          f"extension budget={budget.extension} ========")

    ev = agents.evidence(run_id, batch, budget.n, state.get("critique"))
    rule = ev.rule()
    print(f"[evidence] {rule['weights']}")
    flags = safety(run_id, batch, cells)
    print(f"[safety] {len(flags)} flag(s)")
    spent = 0

    # ---- checkpoint 50 -> 100
    f50 = features.at_checkpoint(run_id, batch, 50)
    r50 = features.rank(features.score(rule, f50))
    opts = options_at_50(budget, len(r50), spent)
    r50_dq = features.rank(features.score(DQ_ONLY, f50))
    ext, pid = checkpoint(run_id, batch, 50, opts, {"rule": r50, "dq_only": r50_dq}, flags, budget.extension - spent,
                          {"n_alive": len(r50), "k_final": budget.k_final, "budget": budget.extension, "spent": spent})
    state["m150_planned"] = next((r["output"] for r in research_log.rows(run_id=run_id, event="checkpoint_plan")
                                  if r["output"]["plan_id"] == pid), {}).get("planned_m150")
    got = visible.materialize(run_id, batch, ext, 50, 100, pid)
    cost = int((got["paid_to"] - 50).sum()) if len(got) else 0
    spent += cost
    research_log.append(run_id, "runner", "extension_done", {"plan_id": pid, "from": 50, "to": 100, "cells": ext,
                        "cost": cost, "died": got[got["eol"].notna()].to_dict("records")}, batch=batch)
    print(f"[ckpt 50] continued {len(ext)} to 100 (cost {cost}); stopped {len(r50) - len(ext)}")

    # ---- checkpoint 100 -> 150
    f100 = features.at_checkpoint(run_id, batch, 100)
    r100 = features.rank(features.score(rule, f100))
    planned = state.get("m150_planned")
    opts = options_at_100(budget, len(r100), spent, planned)
    r100_dq = features.rank(features.score(DQ_ONLY, f100))
    ext, pid = checkpoint(run_id, batch, 100, opts, {"rule": r100, "dq_only": r100_dq}, flags, budget.extension - spent,
                          {"n_alive": len(r100), "k_final": budget.k_final, "budget": budget.extension, "spent": spent})
    got = visible.materialize(run_id, batch, ext, 100, 150, pid)
    cost = int((got["paid_to"] - 100).sum()) if len(got) else 0
    spent += cost
    research_log.append(run_id, "runner", "extension_done", {"plan_id": pid, "from": 100, "to": 150, "cells": ext,
                        "cost": cost, "died": got[got["eol"].notna()].to_dict("records")}, batch=batch)
    print(f"[ckpt 100] continued {len(ext)} to 150 (cost {cost}); total extension spend {spent}/{budget.extension}")

    # ---- checkpoint 150: keep to EOL
    f150 = features.at_checkpoint(run_id, batch, 150)
    rule150 = state.get("revised_rule") or rule
    s150 = features.score(rule150, f150)
    rankings = {"rule": features.rank(s150), "dq_only": features.rank(features.score(DQ_ONLY, f150))}
    opts = options_at_150(budget, len(f150))
    keep, pid = checkpoint(run_id, batch, 150, opts, rankings, flags, 0,
                           {"n_alive": len(f150), "k_final": budget.k_final,
                            "rule_150": rule150.get("fit_on", "none"), "revised": "revised_rule" in state})
    eol = visible.reveal_eol(run_id, batch, keep, pid)
    research_log.append(run_id, "runner", "kept_to_eol", {"plan_id": pid, "cells": keep,
                        "outcomes": eol.to_dict("records")}, batch=batch)
    print(f"[ckpt 150] kept {len(keep)} to end of life")

    # ---- Critic: trigger on revealed kept cells
    rev = eol.merge(f150, on="cell_id").assign(score=lambda d: d["cell_id"].map(s150))
    rho = float(spearmanr(rev["score"], np.log10(rev["cycle_life"].astype(float))).statistic) if len(rev) > 2 else float("nan")
    pre = prereg.load()["revision"]
    fired = bool(rho < 0.5)  # pre-registered: Spearman < 0.5 (NaN does not satisfy it)
    revs_here = sum(r["batch"] == batch for r in research_log.rows(run_id=run_id, event="rule_revised"))
    permitted = batch in pre["allowed_after"] and fired and revs_here < pre["max_per_batch"]
    trig = {"spearman_revealed": rho, "threshold": 0.5, "fired": fired, "n_revealed": len(rev)}
    if role != "test":
        state["pool"] = pd.concat([state.get("pool", pd.DataFrame()), rev], ignore_index=True)
    max_feats = max(1, len(state.get("pool", rev)) // 4)
    crit = agents.critic(run_id, batch, permitted, trig, eol.to_dict("records"), list(rule["weights"]), max_feats)
    new_rule = None
    if crit.revise and permitted:
        feats = ["dq_logvar", *[f for f in crit.new_features if f != "dq_logvar"]][:max_feats]
        new_rule = fit_ridge(state["pool"], feats, fit_on="+".join(sorted(state["pool"]["cell_id"].str[:2].unique())) + "_revealed")
        state["revised_rule"] = new_rule
        research_log.append(run_id, "orchestrator", "rule_revised", new_rule, batch=batch)
    state["critique"] = {"trigger": trig, "revise": new_rule is not None, "reason": crit.reason}
    research_log.append(run_id, "critic_safety", "critique",
                        {**crit.model_dump(), "trigger": trig, "permitted": permitted, "new_rule": new_rule}, batch=batch)
    print(f"[critic] spearman={rho:.2f} fired={fired} permitted={permitted} revised={new_rule is not None}")


def run(run_id: str | None = None, batches: list[str] | None = None) -> str:
    meta = prereg.verify()
    rl = roles()
    batches = batches or [b for b in ("b1", "b2", "b3") if b in rl]
    run_id = run_id or f"v2-{uuid.uuid4().hex[:8]}"
    visible.create(run_id)
    research_log.append(run_id, "orchestrator", "run_start", {
        "version": 2, **meta, "llm_backend": llm.backend(), "model": llm.model_name(),
        "DEMO_AUTO_APPROVE": approval.demo_auto(), "batches": {b: rl[b] for b in batches}})
    if approval.demo_auto():
        print(">>> DEMO_AUTO_APPROVE=true — approvals in this run are automatic and logged as such <<<")
    state: dict = {}
    try:
        for b in batches:
            run_batch(run_id, b, rl[b], state)
    except RunStopped as e:
        print(f"[orchestrator] {e}: stopping cleanly")
        return run_id
    from cyclewise.v2 import scoring
    out = scoring.evaluate(run_id)
    scoring.print_summary(out)
    research_log.append(run_id, "orchestrator", "run_complete", {"log_chain_ok": research_log.verify()})
    return run_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id")
    ap.add_argument("--batches", nargs="+")
    a = ap.parse_args()
    print(f"run_id: {run(a.run_id, a.batches)}")


if __name__ == "__main__":
    main()
