"""Headless First Fifty loop: the same agents, tools, gates, and log as the Omnigent
deployment (omnigent/cyclewise_lab), driven by plain Python so it can run in CI,
offline, and for replay.

  batch 1: Evidence -> Safety scan -> Planner (>=2 options) -> HUMAN APPROVAL
           -> Runner (commit selection, reveal) -> Critic (maybe revise, max 1)
  batch 2: same loop; the revised rule (fit on batch-1 revealed cells only) is
           applied with no tuning on batch 2. Selection committed before reveal.

Run:  python -m cyclewise.run_loop            (interactive approval)
      DEMO_AUTO_APPROVE=true python -m cyclewise.run_loop
"""

from __future__ import annotations

import argparse
import uuid

from cyclewise.agents import approval, critic_safety, evidence, llm, planner, runner
from cyclewise.config import load_config
from cyclewise.data.splits import FROZEN_PATH
from cyclewise.eval import report
from cyclewise.record import research_log
from cyclewise.tools import budget


def banner(msg: str) -> None:
    print(f"\n{'=' * 8} {msg} {'=' * 8}", flush=True)


def preflight(cfg: dict) -> None:
    """Fail before logging anything if the warehouse is not built."""
    from cyclewise.config import path
    need = [FROZEN_PATH, path(cfg["storage"]["early_db"]), path(cfg["storage"]["hidden_db"])]
    gone = [str(p) for p in need if not p.exists()]
    if gone:
        raise SystemExit(f"missing {gone}.\nBuild them first:\n  python -m cyclewise.data.load_raw\n"
                         "  python -m cyclewise.data.splits")


def run(run_id: str | None = None) -> str:
    cfg = load_config()
    run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
    preflight(cfg)
    research_log.append(run_id, "orchestrator", "run_start", {
        "config_hash": cfg["_hash"], "llm_backend": llm.backend(), "model": llm.model_name(),
        "budget_cells": budget.total_budget(), "DEMO_AUTO_APPROVE": approval.demo_auto(),
        "batches": {k: b["role"] for k, b in cfg["batches"].items()},
    })
    if approval.demo_auto():
        print(">>> DEMO_AUTO_APPROVE=true — approvals in this run are automatic and logged as such <<<")

    revised_rule, critique = None, None
    for batch, b in cfg["batches"].items():
        role = b["role"]
        banner(f"BATCH {batch} ({b['name']}, role={role})")

        hyp = evidence.run(run_id, batch, previous_rule=revised_rule, critique=critique)
        rule = revised_rule if revised_rule else hyp.as_rule()
        print(f"[evidence] features: {[(f.name, f.direction) for f in hyp.features]}")
        print(f"[evidence] rule in force: {rule['kind']} (fit_on={rule.get('fit_on')})")

        flags = critic_safety.safety_scan(run_id, batch)
        print(f"[safety] {len(flags)} flag(s)")

        remaining = budget.total_budget()
        plan = planner.run(run_id, batch, role, rule, flags, remaining)
        decision = approval.request(run_id, plan)
        if decision.decision == "rejected":
            print("[human] rejected; planner re-proposes once with the note")
            plan = planner.run(run_id, batch, role, rule, flags, remaining, note=decision.note)
            decision = approval.request(run_id, plan)
            if decision.decision == "rejected":
                research_log.append(run_id, "orchestrator", "run_stopped",
                                    {"reason": "plan rejected twice", "batch": batch}, batch=batch)
                print("[orchestrator] plan rejected twice: stopping the run cleanly")
                return run_id
        if decision.excluded_flagged:
            plan.cell_ids = [c for c in plan.cell_ids if c not in decision.excluded_flagged]
            plan.cost_cells = len(plan.cell_ids)
            research_log.append(run_id, "orchestrator", "flagged_cells_dropped",
                                {"plan_id": plan.plan_id, "dropped": decision.excluded_flagged}, batch=batch)

        result = runner.run(run_id, plan)
        m = result.metrics
        print(f"[runner] revealed {m['n_selected']} cells: {m['hits']} long-lived "
              f"(precision {m['precision_at_k']:.2f}, spearman {m['spearman_revealed']:.2f})")

        crit = critic_safety.run(run_id, result, rule, role, revisions_used=0)
        critique = crit.model_dump()
        print(f"[critic] trigger fired={crit.trigger['fired']}  revise={crit.revise}: {crit.reason}")
        if crit.revise and crit.new_rule:
            revised_rule = crit.new_rule
            research_log.append(run_id, "orchestrator", "rule_revised", revised_rule, batch=batch)
            print(f"[critic] new rule: ridge on {revised_rule['features']} "
                  f"(n={revised_rule['n_train']}, alpha={revised_rule['alpha']})")

    banner("EVALUATION (scoring harness, full labels)")
    summary = report.evaluate(run_id)
    report.print_summary(summary)
    research_log.append(run_id, "orchestrator", "run_complete", {"log_chain_ok": research_log.verify()})
    return run_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id")
    args = ap.parse_args()
    print(f"run_id: {run(args.run_id)}")


if __name__ == "__main__":
    main()
