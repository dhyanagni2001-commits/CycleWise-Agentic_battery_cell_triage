"""Omnigent policies for CycleWise (referenced from omnigent/cyclewise_lab/config.yaml).

Policy callables receive an Omnigent PolicyEvent dict and return
{"result": "ALLOW" | "ASK" | "DENY", "reason": ...} or None (abstain).
See https://omnigent.ai/docs/policies/custom.
"""

from __future__ import annotations

import json
from typing import Any

from cyclewise.config import load_config

HIDDEN_NAMES = ("labels_hidden", "cycles_raw", "hidden.duckdb", "frozen_params", "frozen.yaml")
CYCLE_KEYS = ("cycle", "max_cycle", "min_cycle", "cycle_index", "upto_cycle")


def _args(event: dict) -> dict:
    a = (event.get("data") or {}).get("arguments") or {}
    return json.loads(a) if isinstance(a, str) else a


def _walk(obj: Any):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def leakage_guard(event: dict) -> dict | None:
    """DENY any tool call that asks for a cycle past the cutoff or names a hidden table."""
    if event.get("type") != "tool_call":
        return None
    cutoff = load_config()["preregistered"]["early_cutoff"]
    args = _args(event)
    for k, v in _walk(args):
        if k in CYCLE_KEYS:
            try:
                if int(v) > cutoff:
                    return {"result": "DENY", "reason": f"cycle {v} is past the early cutoff ({cutoff})"}
            except (TypeError, ValueError):
                return {"result": "DENY", "reason": f"non-integer cycle argument {k}={v!r}"}
    blob = json.dumps(args).lower()
    for name in HIDDEN_NAMES:
        if name in blob:
            return {"result": "DENY", "reason": f"reference to hidden data ({name}) is not allowed"}
    # Shell / file tools must not touch the hidden DB either.
    if (event.get("data") or {}).get("name", "").startswith("sys_os") and "warehouse/hidden" in blob:
        return {"result": "DENY", "reason": "hidden warehouse is off limits"}
    return {"result": "ALLOW"}


def approval_gate(event: dict) -> dict | None:
    """ASK a human before any plan is approved. The reason shows the plan."""
    name = (event.get("data") or {}).get("name")
    if event.get("type") != "tool_call" or name not in ("request_approval", "v2_request_approval"):
        return None
    args = _args(event)
    try:
        if name == "v2_request_approval":
            text = _render_v2(args["run_id"], args["plan_id"])
        else:
            from cyclewise.agents.approval import render
            from cyclewise.agents.planner import load_plan
            text = render(load_plan(args["run_id"], args["plan_id"]))
    except Exception as e:  # still ASK; never auto-approve
        text = f"plan {args.get('plan_id')} (could not render: {e})"
    return {"result": "ASK", "reason": text}


def _render_v2(run_id: str, plan_id: str) -> str:
    from cyclewise.record import research_log
    from cyclewise.v2.loop import render
    p = [r["output"] for r in research_log.rows(run_id=run_id, event="checkpoint_plan")
         if r["output"]["plan_id"] == plan_id][-1]
    chosen = next(o for o in p["options"] if o["option_id"] == p["chosen"])
    return render(p["batch"], p["checkpoint"], plan_id, chosen, p["options"], p["reason"], p["selected"],
                  p["stopped"], p["flags"], f"Remaining extension budget: {p['remaining_budget']} channel-cycles")


def reveal_gate(event: dict) -> dict | None:
    """DENY execute_plan unless an approval for that exact plan is in the research log."""
    if event.get("type") != "tool_call" or (event.get("data") or {}).get("name") not in ("execute_plan",
                                                                                      "v2_execute_checkpoint"):
        return None
    args = _args(event)
    from cyclewise.record import research_log
    ok = any(r["output"].get("plan_id") == args.get("plan_id") and r["output"].get("decision") == "approved"
             for r in research_log.rows(run_id=args.get("run_id"), event="approval"))
    if not ok:
        return {"result": "DENY", "reason": f"plan {args.get('plan_id')} has no human approval on record"}
    return {"result": "ALLOW"}


def budget_guard(event: dict) -> dict | None:
    """DENY plan submissions whose options exceed the cell budget."""
    if event.get("type") != "tool_call" or (event.get("data") or {}).get("name") != "submit_test_plan":
        return None
    args = _args(event)
    k = load_config()["preregistered"]["budget_cells"]
    try:
        draft = json.loads(args["plan_draft"]) if isinstance(args.get("plan_draft"), str) else args.get("plan_draft")
        over = [o for o in draft.get("options", []) if int(o.get("n_cells", 0)) > k]
    except Exception:
        return None  # malformed: the tool's own validation will reject it
    if over:
        return {"result": "DENY", "reason": f"option(s) {[o.get('option_id') for o in over]} exceed {k} cells"}
    return {"result": "ALLOW"}


POLICY_REGISTRY = [
    {"handler": "cyclewise.policies.omni_policies.leakage_guard", "kind": "function",
     "name": "CycleWise leakage guard", "description": "Deny cycle > 50 and hidden-table access."},
    {"handler": "cyclewise.policies.omni_policies.approval_gate", "kind": "function",
     "name": "CycleWise approval gate", "description": "Human must approve every test plan."},
    {"handler": "cyclewise.policies.omni_policies.reveal_gate", "kind": "function",
     "name": "CycleWise reveal gate", "description": "No reveal without a recorded approval."},
    {"handler": "cyclewise.policies.omni_policies.budget_guard", "kind": "function",
     "name": "CycleWise budget guard", "description": "Deny plans over the cell budget."},
]
