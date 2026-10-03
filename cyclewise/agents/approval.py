"""Human approval gate. Nothing is revealed without an `approval` row in the log.

Modes:
  interactive  stdin is a TTY: show the plan, ask approve / reject + note, and which
               flagged cells to drop.
  file         no TTY: write warehouse/approvals/<plan_id>.request.json and WAIT (no
               timeout, no auto-approve) until a decision is recorded with
               `python -m cyclewise.agents.approval <plan_id> approve|reject [note]`.
  demo         only if config approval.demo_auto_approve is true or env
               DEMO_AUTO_APPROVE=true; the approval row and the screen both say
               DEMO_AUTO_APPROVE=true.

In Omnigent the same gate is a policy that returns ASK on the `approve_plan` /
`reveal_outcomes` tool calls (cyclewise/policies/omni_policies.py).
"""

from __future__ import annotations

import json
import os
import sys
import time

from cyclewise.agents.schemas import Approval, TestPlan
from cyclewise.config import load_config, path
from cyclewise.record import research_log

APPROVALS = "warehouse/approvals"


def demo_auto() -> bool:
    return bool(load_config()["approval"]["demo_auto_approve"]) or os.environ.get("DEMO_AUTO_APPROVE") == "true"


def render(plan: TestPlan) -> str:
    lines = [f"=== APPROVAL REQUEST  plan {plan.plan_id}  batch {plan.batch} ===",
             f"Chosen option {plan.chosen}: {plan.reason}",
             f"Cost: {plan.cost_cells} cells (budget unit: cells)"]
    for o in plan.options:
        mark = "*" if o.option_id == plan.chosen else " "
        lines.append(f" {mark} [{o.option_id}] {o.strategy}: {o.description}  | learning: {o.expected_learning}")
    lines.append(f"Cells to keep testing: {', '.join(plan.cell_ids)}")
    if plan.tie_at_boundary:
        lines.append("NOTE: score tie at the budget boundary, broken by cell id")
    if plan.flagged_cells:
        lines.append("Safety flags:")
        for f in plan.flagged_cells:
            sel = "SELECTED" if f["cell_id"] in plan.cell_ids else "not selected"
            lines.append(f"   {f['cell_id']} [{sel}] {f['flag']}: {f['detail']}")
    else:
        lines.append("Safety flags: none")
    return "\n".join(lines)


def request(run_id: str, plan: TestPlan) -> Approval:
    """v1: approve a TestPlan."""
    flagged_sel = [f["cell_id"] for f in plan.flagged_cells if f["cell_id"] in plan.cell_ids]
    return decide(run_id, plan.plan_id, plan.batch, render(plan), plan.model_dump_json(indent=2), flagged_sel)


def decide(run_id: str, plan_id: str, batch: str, text: str, payload_json: str,
           flagged_in_selection: list[str] | None = None) -> Approval:
    """Ask a human to approve one decision. Shared by v1 plans and v2 checkpoints.

    demo mode (labeled) -> interactive TTY -> otherwise wait for a decision file, forever.
    """
    print(text, flush=True)
    flagged_in_selection = flagged_in_selection or []
    if demo_auto():
        a = Approval(plan_id=plan_id, decision="approved", approver="DEMO_AUTO_APPROVE",
                     note="DEMO_AUTO_APPROVE=true", demo_auto_approve=True)
        print(">>> DEMO_AUTO_APPROVE=true: plan auto-approved for the demo <<<", flush=True)
    elif sys.stdin.isatty():
        try:
            a = _interactive(plan_id, flagged_in_selection)
        except (EOFError, KeyboardInterrupt):
            research_log.append(run_id, "human", "approval_aborted", {"plan_id": plan_id}, batch=batch)
            raise SystemExit("\nNo decision given: nothing approved, nothing revealed. Run stopped.")
    else:
        a = _wait_for_file(plan_id, payload_json)
    research_log.append(run_id, "human", "approval", a.model_dump(), batch=batch, inputs=plan_id)
    return a


def _interactive(plan_id: str, flagged_in_selection: list[str]) -> Approval:
    ans = input("Approve this plan? [y/N]: ").strip().lower()
    note = input("Note for the planner (optional): ").strip()
    drop: list[str] = []
    if ans == "y" and flagged_in_selection:
        d = input("Drop any flagged cells from the selection? ids comma-separated, blank = keep all: ").strip()
        drop = [x.strip() for x in d.split(",") if x.strip() in flagged_in_selection]
    return Approval(plan_id=plan_id, decision="approved" if ans == "y" else "rejected",
                    approver=os.environ.get("USER", "human"), note=note, excluded_flagged=drop)


def _wait_for_file(plan_id: str, payload_json: str) -> Approval:
    d = path(APPROVALS)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{plan_id}.request.json").write_text(payload_json)
    decision = d / f"{plan_id}.decision.json"
    print(f"Waiting for a human decision. Run:\n  python -m cyclewise.agents.approval {plan_id} approve "
          f"\"note\"\n  python -m cyclewise.agents.approval {plan_id} reject \"note\"", flush=True)
    while not decision.exists():
        time.sleep(2)
    return Approval(**json.loads(decision.read_text()))


def main() -> None:
    plan_id, verdict = sys.argv[1], sys.argv[2]
    note = sys.argv[3] if len(sys.argv) > 3 else ""
    a = Approval(plan_id=plan_id, decision="approved" if verdict == "approve" else "rejected",
                 approver=os.environ.get("USER", "human"), note=note)
    out = path(APPROVALS) / f"{plan_id}.decision.json"
    out.write_text(a.model_dump_json())
    print(f"recorded {a.decision} for {plan_id}")


if __name__ == "__main__":
    main()
