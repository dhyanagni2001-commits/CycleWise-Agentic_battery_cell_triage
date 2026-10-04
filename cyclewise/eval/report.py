"""Evaluate a finished run against both baselines, with bootstrap CIs.

Reads the committed selections from the research log, scores them with the
hidden labels under both pre-registered label rules, logs baselines to MLflow,
and writes reports/<run_id>.json/.md.
"""

from __future__ import annotations

import json
import math

from cyclewise.baselines import delta_q_model, early_capacity
from cyclewise.config import ROOT, load_config
from cyclewise.eval import bootstrap, metrics, tracking
from cyclewise.record import research_log
from cyclewise.tools.cutoff_view import list_cells
from cyclewise.tools.train_eval import score, select_top_k

REPORTS = ROOT / "reports"
BASELINES = ("early_capacity", "delta_q")
RULE_TEXT = {
    "primary_train_q75": "long-lived = cycle life >= 75th percentile of batch-1 (train) cycle life, frozen before any run",
    "secondary_within_batch_q75": "long-lived = cycle life >= 75th percentile within the same batch (pre-registered sensitivity check)",
}


def _fmt(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"


def _score(lab, sels: dict, ranks: dict, k: int, batch: str, run_id: str, log_mlflow: bool) -> dict:
    n_pos = int(lab["long_lived"].astype(bool).sum())
    r = {"n_scored": len(lab), "n_long_lived": n_pos,
         "random_expected_recall": min(1.0, k / len(lab)) if len(lab) else float("nan"),
         "oracle_recall": min(1.0, k / n_pos) if n_pos else float("nan"), "methods": {}}
    if n_pos == 0:
        r["undefined"] = "no long-lived cells under this rule: recall is undefined"
        for m, s in sels.items():
            r["methods"][m] = {"recall": float("nan"), "precision": metrics.precision_at_k(s, lab),
                               "n_selected": len(s), "selected": sorted(s)}
        return r
    cis = bootstrap.recall_ci(sels, lab, reference="cyclewise")
    for m, s in sels.items():
        rec = metrics.recall_at_k(s, lab)
        r["methods"][m] = {"recall": rec, "precision": metrics.precision_at_k(s, lab),
                           "n_selected": len(s), "ci_low": cis[m]["ci_low"], "ci_high": cis[m]["ci_high"],
                           "selected": sorted(s)}
        if log_mlflow and m != "cyclewise":
            tracking.log_selection(f"{m}_{batch}", batch, m, {"k": k, "run_id": run_id},
                                   {"recall_at_k": rec, "precision_at_k": r["methods"][m]["precision"]}, s)
    for m in BASELINES:  # bootstrap gives baseline - cyclewise; flip the sign
        d = cis[m]["diff_vs_cyclewise"]
        r["methods"]["cyclewise"][f"diff_vs_{m}"] = {"mean": -d["mean"], "ci_low": -d["ci_high"],
                                                     "ci_high": -d["ci_low"]}
    best = max(BASELINES, key=lambda m: r["methods"][m]["recall"])
    target = r["methods"][best]["recall"]
    accel = {}
    for m in ("cyclewise", *BASELINES):
        kk = metrics.cells_to_reach(ranks[m], lab, target)
        accel[m] = {"cells_needed": kk, "test_cycles_needed": metrics.cycles_for(ranks[m], kk, lab)}
    cw, bb = accel["cyclewise"]["cells_needed"], accel[best]["cells_needed"]
    r["acceleration"] = {"best_baseline": best, "target_recall": target, **accel,
                         "cells_ratio_best_over_cyclewise": (bb / cw) if cw and bb else None,
                         "note": "testing cells in each rule's full rank order until the best baseline's recall@K is matched"}
    return r


def evaluate(run_id: str) -> dict:
    cfg = load_config()
    k = cfg["preregistered"]["budget_cells"]
    labels = metrics.load_labels()
    log = research_log.rows(run_id=run_id)
    plans = {r["output"]["plan_id"]: r["output"] for r in log if r["event"] == "test_plan"}
    commits = {r["batch"]: r["output"] for r in log if r["event"] == "selection_committed"}
    run_start = next((r["output"] for r in log if r["event"] == "run_start"), {})
    out = {"run_id": run_id, "budget_cells": k, "llm_backend": run_start.get("llm_backend"),
           "model": run_start.get("model"), "demo_auto_approve": run_start.get("DEMO_AUTO_APPROVE"),
           "fallback_steps": sum(bool(r["fallback_used"]) for r in log),
           "agent_steps": sum(r["agent"] in ("evidence", "planner", "critic_safety") and r["event"] in
                              ("hypothesis", "test_plan", "critique") for r in log),
           "revised": any(r["event"] == "rule_revised" for r in log), "batches": {}}

    for batch, b in cfg["batches"].items():
        if batch not in commits:
            continue
        eligible = list_cells(batch)["cell_id"].tolist()
        plan = plans[commits[batch]["plan_id"]]
        sels = {
            "cyclewise": commits[batch]["cell_ids"],
            "early_capacity": early_capacity.select(batch, k, eligible)["cell_ids"],
            "delta_q": delta_q_model.select(batch, k, eligible)["cell_ids"],
        }
        ranks = {
            "cyclewise": select_top_k(score(plan["rule"], batch, eligible), len(eligible), eligible)["ranking"],
            "early_capacity": early_capacity.select(batch, len(eligible), eligible)["ranking"],
            "delta_q": delta_q_model.select(batch, len(eligible), eligible)["ranking"],
        }
        if b["role"] == "eval":
            train = [kk for kk, bb in cfg["batches"].items() if bb["role"] == "train"][0]
            ref = delta_q_model.fit_supervised(train, labels)
            sels["ref_delta_q_all_train_labels"] = select_top_k(score(ref, batch, eligible), k, eligible)["cell_ids"]
        res = {"n_cells": len(eligible),
               "n_censored": int(labels[labels["batch"] == batch]["censored"].sum()),
               "n_label_unknown_primary": int((~labels[labels["batch"] == batch]["label_known"]).sum()),
               "n_flagged": len(plan.get("flagged_cells", [])),
               "strategy": plan["chosen"], "rule": plan["rule"], "label_rules": {}}
        for name, rlab in (("primary_train_q75", metrics.scored(labels, batch)),
                           ("secondary_within_batch_q75", metrics.within_batch_labels(labels, batch))):
            res["label_rules"][name] = _score(rlab, sels, ranks, k, batch, run_id, name.startswith("primary"))
        out["batches"][batch] = res

    REPORTS.mkdir(exist_ok=True)
    for name in (f"{run_id}.json", "latest.json"):
        (REPORTS / name).write_text(json.dumps(out, indent=2, default=str))
    (REPORTS / f"{run_id}.md").write_text(to_markdown(out))
    research_log.append(run_id, "evaluator", "evaluation", {
        b: {rn: {m: {kk: v for kk, v in mv.items() if kk != "selected"} for m, mv in rr["methods"].items()}
            for rn, rr in res["label_rules"].items()} for b, res in out["batches"].items()})
    return out


def to_markdown(out: dict) -> str:
    lines = [f"# First Fifty results: {out['run_id']}", "",
             f"LLM backend: {out['llm_backend']} ({out['model']}); deterministic-fallback steps: "
             f"{out['fallback_steps']}; DEMO_AUTO_APPROVE={out['demo_auto_approve']}; rule revised: {out['revised']}.",
             f"Budget: {out['budget_cells']} cells per batch. Primary metric: recall of long-lived cells at that "
             "budget, 95% bootstrap CIs (cells resampled with replacement, fixed seed).", ""]
    for batch, r in out["batches"].items():
        lines += [f"## Batch {batch}", "",
                  f"{r['n_cells']} eligible cells; {r['n_censored']} censored ({r['n_label_unknown_primary']} with "
                  f"unknown label under the primary rule, excluded from scoring); {r['n_flagged']} safety flags. "
                  f"First Fifty chose option {r['strategy']} with rule `{r['rule']['kind']}` "
                  f"(fit_on={r['rule'].get('fit_on')}).", ""]
        for rn, rr in r["label_rules"].items():
            lines += [f"### {rn}", "", f"_{RULE_TEXT[rn]}._ {rr['n_long_lived']} long-lived of {rr['n_scored']} scored.", ""]
            if rr.get("undefined"):
                lines += [f"**{rr['undefined']}.** Precision of each selection: "
                          + ", ".join(f"{m} {_fmt(v['precision'])}" for m, v in rr["methods"].items()), ""]
                continue
            lines += ["| method | recall@K | 95% CI | precision@K |", "|---|---|---|---|"]
            for m, v in rr["methods"].items():
                lines.append(f"| {m} | {_fmt(v['recall'])} | [{_fmt(v['ci_low'])}, {_fmt(v['ci_high'])}] | {_fmt(v['precision'])} |")
            lines += [f"| random (expected) | {_fmt(rr['random_expected_recall'])} | | |",
                      f"| oracle (max possible) | {_fmt(rr['oracle_recall'])} | | |", ""]
            cw = rr["methods"]["cyclewise"]
            for m in BASELINES:
                d = cw[f"diff_vs_{m}"]
                lines.append(f"- First Fifty minus {m}: {d['mean']:+.2f} recall (95% CI [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}])")
            a = rr["acceleration"]
            ratio = a["cells_ratio_best_over_cyclewise"]
            lines += [f"- Cells tested in rank order to match {a['best_baseline']} (recall {_fmt(a['target_recall'])}): "
                      + ", ".join(f"{m} {a[m]['cells_needed']}" for m in ("cyclewise", *BASELINES))
                      + (f". Ratio = {ratio:.2f}x" if ratio else ""), ""]
    return "\n".join(lines)


def print_summary(out: dict) -> None:
    for batch, r in out["batches"].items():
        for rn, rr in r["label_rules"].items():
            print(f"\nBatch {batch} [{rn}]: {rr['n_long_lived']} long-lived of {rr['n_scored']} scored")
            if rr.get("undefined"):
                print(f"  {rr['undefined']}")
                continue
            for m, v in rr["methods"].items():
                print(f"  {m:30s} recall {_fmt(v['recall'])}  CI [{_fmt(v['ci_low'])}, {_fmt(v['ci_high'])}]")
            a = rr["acceleration"]
            print(f"  cells to match {a['best_baseline']}: "
                  + ", ".join(f"{m}={a[m]['cells_needed']}" for m in ("cyclewise", *BASELINES)))
