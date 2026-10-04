"""Omnigent tools for First Fifty (sequential checkpoints at 50/100/150).

Referenced by the generated files in omnigent/cyclewise_v2/**/tools/python/.
All run state (rule in force, budget spent, next checkpoint, revised rule) is
derived from the research log, never trusted from tool arguments. The order
is enforced: hypothesis + safety -> checkpoint 50 -> 100 -> 150 -> critique,
batch after batch. Every tool returns {"ok": false, "error": ...} on bad input.
"""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cyclewise import prereg
from cyclewise.agents import citations
from cyclewise.agents.schemas import Approval
from cyclewise.record import research_log
from cyclewise.tools.omni_tools import _parse, safe
from cyclewise.v2 import features, visible
from cyclewise.v2 import scoring
from cyclewise.v2.agents import CritiqueV2, EvidenceV2, PROMPT_VERSION
from cyclewise.v2.allocation import Budget, cut_to_budget, options_at_50, options_at_100, options_at_150
from cyclewise.v2.loop import DQ_ONLY, fit_ridge, render, roles, safety

ORDER = ("b1", "b2", "b3")


# ------------------------------------------------------------------ state from the log

def _rows(run_id: str, event: str, batch: str | None = None) -> list[dict]:
    return research_log.rows(run_id=run_id, event=event, batch=batch)


def _require_run(run_id: str) -> dict:
    start = _rows(run_id, "run_start")
    if not start or start[-1]["output"].get("version") != 2:
        raise ValueError(f"unknown v2 run {run_id!r}; call v2_start_run first")
    return start[-1]["output"]


def _require_batch(run_id: str, batch: str) -> str:
    start = _require_run(run_id)
    if batch not in start["batches"]:
        raise ValueError(f"batch {batch!r} is not in this run; batches: {list(start['batches'])}")
    return start["batches"][batch]


def _budget(run_id: str, batch: str) -> Budget:
    return Budget.for_batch(len(visible.paid(run_id, batch)))


def _rule(run_id: str, batch: str) -> dict:
    h = _rows(run_id, "hypothesis", batch)
    if not h:
        raise ValueError(f"no hypothesis for {batch} yet; the Evidence agent must call v2_submit_hypothesis")
    return EvidenceV2.model_validate(h[-1]["output"]).rule()


def _revised_before(run_id: str, batch: str) -> dict | None:
    earlier = set(ORDER[: ORDER.index(batch)])
    rows = [r for r in _rows(run_id, "rule_revised") if r["batch"] in earlier]
    return rows[-1]["output"] if rows else None


def _flags(run_id: str, batch: str) -> list[dict]:
    f = _rows(run_id, "safety_flags", batch)
    if not f:
        raise ValueError(f"no safety scan for {batch} yet; call v2_safety_scan first")
    return f[-1]["output"]


def _spent(run_id: str, batch: str) -> int:
    return sum(r["output"]["cost"] for r in _rows(run_id, "extension_done", batch))


def _executed(run_id: str, batch: str) -> list[int]:
    return sorted(r["output"]["checkpoint"] for r in _rows(run_id, "checkpoint_committed", batch))


def _next_checkpoint(run_id: str, batch: str) -> int | None:
    done = _executed(run_id, batch)
    for c in (50, 100, 150):
        if c not in done:
            return c
    return None


def _check_batch_order(run_id: str, batch: str) -> None:
    start = _require_run(run_id)
    for prev in [b for b in ORDER[: ORDER.index(batch)] if b in start["batches"]]:
        if not _rows(run_id, "critique", prev):
            raise ValueError(f"finish batch {prev} (all checkpoints and the critique) before starting {batch}")


def _rankings(run_id: str, batch: str, ckpt: int) -> tuple[dict, int, dict]:
    rule = _rule(run_id, batch)
    if ckpt == 150:
        rule = _revised_before(run_id, batch) or rule
    f = features.at_checkpoint(run_id, batch, ckpt)
    rk = {"rule": features.rank(features.score(rule, f)), "dq_only": features.rank(features.score(DQ_ONLY, f))}
    return rk, len(f), rule


def _options(run_id: str, batch: str, ckpt: int, n_alive: int) -> list[dict]:
    b, spent = _budget(run_id, batch), _spent(run_id, batch)
    if ckpt == 50:
        return options_at_50(b, n_alive, spent)
    if ckpt == 100:
        p50 = [r["output"] for r in _rows(run_id, "checkpoint_plan", batch) if r["output"]["checkpoint"] == 50
               and r["output"]["plan_id"] in {c["output"]["plan_id"] for c in _rows(run_id, "checkpoint_committed", batch)}]
        return options_at_100(b, n_alive, spent, p50[-1].get("planned_m150") if p50 else None)
    return options_at_150(b, n_alive)


def _plan(run_id: str, plan_id: str) -> dict:
    rows = [r["output"] for r in _rows(run_id, "checkpoint_plan") if r["output"]["plan_id"] == plan_id]
    if not rows:
        raise KeyError(f"unknown plan {plan_id!r} in run {run_id}")
    return rows[-1]


# ------------------------------------------------------------------ supervisor tools

@safe
def v2_start_run() -> dict:
    """Start a First Fifty run (checkpoints 50/100/150). Returns run_id, batches with roles, budgets."""
    meta = prereg.verify()
    rl = roles()
    run_id = f"omni2-{uuid.uuid4().hex[:8]}"
    visible.create(run_id)
    batches = {b: rl[b] for b in ORDER if b in rl}
    research_log.append(run_id, "orchestrator", "run_start", {
        "version": 2, **meta, "orchestrator": "omnigent", "llm_backend": "omnigent (claude-sdk harness)",
        "model": "Claude via Omnigent claude-sdk harness", "DEMO_AUTO_APPROVE": False, "batches": batches})
    out = []
    for b, role in batches.items():
        bud = _budget(run_id, b)
        out.append({"batch": b, "role": role, "n_cells": bud.n, "k_final": bud.k_final,
                    "extension_budget_channel_cycles": bud.extension})
    return {"ok": True, "run_id": run_id, "batches": out, "prereg_commit": meta["prereg_commit"]}


@safe
def v2_run_state(run_id: str) -> dict:
    """Where the run is: per batch, the next checkpoint, budget spent, revised rule, and the event list."""
    start = _require_run(run_id)
    state = {}
    for b in start["batches"]:
        state[b] = {"role": start["batches"][b], "has_hypothesis": bool(_rows(run_id, "hypothesis", b)),
                    "has_safety_scan": bool(_rows(run_id, "safety_flags", b)),
                    "checkpoints_done": _executed(run_id, b), "next_checkpoint": _next_checkpoint(run_id, b),
                    "extension_spent": _spent(run_id, b), "critique_done": bool(_rows(run_id, "critique", b))}
    rev = _rows(run_id, "rule_revised")
    return {"ok": True, "batches": state, "revised_rule": rev[-1]["output"] if rev else None}


@safe
def v2_request_approval(run_id: str, plan_id: str, note: str = "") -> dict:
    """Record human approval of a checkpoint plan. Gated by an Omnigent ASK policy: this tool only
    runs after a human approves the prompt in the Omnigent UI."""
    p = _plan(run_id, plan_id)
    a = Approval(plan_id=plan_id, decision="approved", approver="omnigent-ask", note=note)
    research_log.append(run_id, "human", "approval", a.model_dump(), batch=p["batch"], inputs=plan_id)
    return {"ok": True, "approval": a.model_dump()}


@safe
def v2_evaluate_run(run_id: str) -> dict:
    """Score the run against every pre-registered comparator (recall, channel-cycles, CIs, claims)."""
    _require_run(run_id)
    out = scoring.evaluate(run_id)
    research_log.append(run_id, "orchestrator", "run_complete", {"log_chain_ok": research_log.verify()})
    return {"ok": True, "markdown": scoring.to_markdown(out)}


# ------------------------------------------------------------------ evidence tools

@safe
def v2_feature_catalog() -> dict:
    """Checkpoint feature catalog (computed at 50, 100, 150) and the allowed citation ids."""
    return {"ok": True, "features": features.FEATURES, "citations": sorted(citations.allowed()),
            "pre_registered": "dq_logvar (direction -1) must be the base signal with the largest weight"}


@safe
def v2_submit_hypothesis(run_id: str, batch: str, hypothesis: str) -> dict:
    """Validate and record the batch's scoring rule: JSON {features:[{name,direction,weight}], rationale,
    citations, label:"agent-generated"}; dq_logvar must be the base signal."""
    _require_batch(run_id, batch)
    _check_batch_order(run_id, batch)
    if _executed(run_id, batch):
        raise ValueError(f"{batch} has already started its checkpoints; the rule cannot change now")
    h = EvidenceV2.model_validate(_parse(hypothesis))
    citations.validate(h.citations)
    research_log.append(run_id, "evidence", "hypothesis", h.model_dump(), batch=batch, model="omnigent",
                        prompt_version=PROMPT_VERSION)
    return {"ok": True, "rule": h.rule()}


# ------------------------------------------------------------------ safety / critic tools

@safe
def v2_safety_scan(run_id: str, batch: str) -> dict:
    """Flag hot cells, IR jumps and missing sensors in cycles 2..50. Flags go to the human at approval."""
    _require_batch(run_id, batch)
    _check_batch_order(run_id, batch)
    cells = visible.paid(run_id, batch)["cell_id"].tolist()
    return {"ok": True, "flags": safety(run_id, batch, cells)}


@safe
def v2_submit_critique(run_id: str, batch: str, critique: str) -> dict:
    """After checkpoint 150: JSON {revise, reason, new_features, next_experiment}. Revision is refused
    unless the pre-registered trigger (Spearman < 0.5 on revealed cells) fired on a b1/b2 batch."""
    role = _require_batch(run_id, batch)
    kept = _rows(run_id, "kept_to_eol", batch)
    if not kept:
        raise ValueError(f"{batch} has not finished checkpoint 150 yet")
    if _rows(run_id, "critique", batch):
        raise ValueError(f"{batch} already has a critique")
    pre = prereg.load()["revision"]
    eol = pd.DataFrame(kept[-1]["output"]["outcomes"])
    f150 = features.at_checkpoint(run_id, batch, 150, eol["cell_id"].tolist())
    rule150 = _revised_before(run_id, batch) or _rule(run_id, batch)   # the rule that ranked the 150 decision
    s150 = features.score(rule150, f150)
    rev = eol[eol["cell_id"].isin(s150.index)]
    rho = (float(spearmanr(s150.loc[rev["cell_id"]].values, np.log10(rev["cycle_life"].astype(float))).statistic)
           if len(rev) > 2 else float("nan"))
    fired = bool(rho < 0.5)
    used = len(_rows(run_id, "rule_revised", batch))
    permitted = batch in pre["allowed_after"] and fired and used < pre["max_per_batch"]
    pool_frames = []
    for b in ORDER[: ORDER.index(batch) + 1]:
        if _require_run(run_id)["batches"].get(b) in ("train", "validation"):
            k = _rows(run_id, "kept_to_eol", b)
            if k:
                e = pd.DataFrame(k[-1]["output"]["outcomes"])
                pool_frames.append(e.merge(features.at_checkpoint(run_id, b, 150, e["cell_id"].tolist()), on="cell_id"))
    pool = pd.concat(pool_frames, ignore_index=True) if pool_frames else pd.DataFrame()
    max_feats = max(1, len(pool) // 4)
    d = CritiqueV2.model_validate(_parse(critique))
    trig = {"spearman_revealed": rho, "threshold": 0.5, "fired": fired, "n_revealed": len(rev), "role": role}
    if d.revise and not permitted:
        return {"ok": False, "error": "revision not permitted: the trigger did not fire, or this is the test batch",
                "trigger": trig}
    new_rule = None
    if d.revise:
        feats = ["dq_logvar", *[f for f in d.new_features if f != "dq_logvar"]][:max_feats]
        new_rule = fit_ridge(pool, feats, fit_on="+".join(sorted(pool["cell_id"].str[:2].unique())) + "_revealed")
        research_log.append(run_id, "orchestrator", "rule_revised", new_rule, batch=batch)
    research_log.append(run_id, "critic_safety", "critique",
                        {**d.model_dump(), "trigger": trig, "permitted": permitted, "new_rule": new_rule},
                        batch=batch, model="omnigent", prompt_version=PROMPT_VERSION)
    return {"ok": True, "trigger": trig, "permitted": permitted, "revised": new_rule is not None, "new_rule": new_rule}


# ------------------------------------------------------------------ planner tools

@safe
def v2_checkpoint_options(run_id: str, batch: str) -> dict:
    """The options for the batch's NEXT checkpoint (50, 100 or 150): all within budget, >= 2 of them,
    plus the top of the current ranking and the remaining budget."""
    _require_batch(run_id, batch)
    _check_batch_order(run_id, batch)
    _rule(run_id, batch)
    _flags(run_id, batch)
    ckpt = _next_checkpoint(run_id, batch)
    if ckpt is None:
        raise ValueError(f"{batch} has finished all checkpoints")
    rk, n_alive, rule = _rankings(run_id, batch, ckpt)
    b = _budget(run_id, batch)
    return {"ok": True, "checkpoint": ckpt, "n_alive": n_alive, "k_final": b.k_final,
            "extension_budget": b.extension, "spent": _spent(run_id, batch),
            "rule_in_force": rule, "options": _options(run_id, batch, ckpt, n_alive),
            "top_by_rule": rk["rule"][:15], "safety_flags": _flags(run_id, batch)}


@safe
def v2_submit_choice(run_id: str, batch: str, option_id: str, reason: str) -> dict:
    """Choose one option for the next checkpoint. Cell ids come from the ranking, never from the agent.
    Returns the plan (plan_id, cells to continue/keep, cells stopped) for the supervisor to approve."""
    _require_batch(run_id, batch)
    _check_batch_order(run_id, batch)
    ckpt = _next_checkpoint(run_id, batch)
    if ckpt is None:
        raise ValueError(f"{batch} has finished all checkpoints")
    if len(reason.strip()) < 10:
        raise ValueError("give a reason of at least 10 characters")
    rk, n_alive, _ = _rankings(run_id, batch, ckpt)
    opts = _options(run_id, batch, ckpt, n_alive)
    chosen = next((o for o in opts if o["option_id"] == option_id), None)
    if chosen is None:
        raise ValueError(f"option {option_id!r} is not one of {[o['option_id'] for o in opts]}")
    remaining = _budget(run_id, batch).extension - _spent(run_id, batch)
    ranked = rk[chosen["ranking"]]
    selected, was_cut = (ranked[: chosen["m"]], False) if ckpt == 150 else cut_to_budget(ranked, chosen["m"], remaining)
    flags = _flags(run_id, batch)
    plan = {"plan_id": f"{batch}-c{ckpt}-{uuid.uuid4().hex[:6]}", "batch": batch, "checkpoint": ckpt,
            "options": opts, "planned_m150": chosen.get("m150_planned"), "chosen": option_id, "reason": reason,
            "selected": selected, "stopped": [c for c in ranked if c not in selected], "cut_at_budget": was_cut,
            "remaining_budget": remaining, "flags": flags}
    research_log.append(run_id, "planner", "checkpoint_plan", plan, batch=batch, model="omnigent",
                        prompt_version=PROMPT_VERSION)
    summary = render(batch, ckpt, plan["plan_id"], chosen, opts, reason, selected, plan["stopped"], flags,
                     f"Remaining extension budget: {remaining} channel-cycles")
    return {"ok": True, "plan_id": plan["plan_id"], "checkpoint": ckpt, "selected": selected,
            "n_stopped": len(plan["stopped"]), "summary_for_human": summary}


# ------------------------------------------------------------------ runner tools

@safe
def v2_execute_checkpoint(run_id: str, plan_id: str) -> dict:
    """Commit an APPROVED checkpoint plan and pay for it: extend the selected cells to the next
    checkpoint (50->100, 100->150), or at 150 keep them to end of life and reveal their outcomes."""
    p = _plan(run_id, plan_id)
    batch, ckpt = p["batch"], p["checkpoint"]
    if ckpt in _executed(run_id, batch):
        raise PermissionError(f"{batch} checkpoint {ckpt} was already executed")
    if _next_checkpoint(run_id, batch) != ckpt:
        raise PermissionError(f"{batch}: checkpoint {_next_checkpoint(run_id, batch)} comes next, not {ckpt}")
    appr = [r["output"] for r in _rows(run_id, "approval") if r["output"].get("plan_id") == plan_id]
    if not appr or appr[-1].get("decision") != "approved":
        raise PermissionError(f"plan {plan_id} has no human approval on record")
    sel = p["selected"]
    key = "keep" if ckpt == 150 else "extend"
    research_log.append(run_id, "runner", "checkpoint_committed",
                        {"plan_id": plan_id, "checkpoint": ckpt, key: sorted(sel),
                         ("extend" if key == "keep" else "keep"): [], "dropped_flagged": []}, batch=batch)
    if ckpt < 150:
        got = visible.materialize(run_id, batch, sel, ckpt, ckpt + 50, plan_id)
        cost = int((got["paid_to"] - ckpt).sum()) if len(got) else 0
        died = got[got["eol"].notna()].to_dict("records")
        research_log.append(run_id, "runner", "extension_done", {"plan_id": plan_id, "from": ckpt, "to": ckpt + 50,
                            "cells": sel, "cost": cost, "died": died}, batch=batch)
        return {"ok": True, "checkpoint": ckpt, "extended": len(sel), "cost_channel_cycles": cost,
                "reached_eol_in_window": died}
    eol = visible.reveal_eol(run_id, batch, sel, plan_id)
    research_log.append(run_id, "runner", "kept_to_eol", {"plan_id": plan_id, "cells": sel,
                        "outcomes": eol.to_dict("records")}, batch=batch)
    return {"ok": True, "checkpoint": 150, "kept_to_eol": len(sel), "outcomes": eol.to_dict("records")}

