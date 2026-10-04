"""Omnigent function tools (referenced by `callable:` in omnigent/cyclewise_lab/**/config.yaml).

Each tool takes JSON-friendly arguments and returns a JSON-serialisable dict. The
LLM sub-agents hand off by calling the `submit_*` tools, which validate against
the Pydantic schemas in cyclewise.agents.schemas and append to the research log.
A validation failure returns {"ok": false, "error": ...} so the agent can retry.
"""

from __future__ import annotations

import functools
import json
import logging
import uuid
from typing import Any

from pydantic import ValidationError

from cyclewise.agents import citations, critic_safety, evidence, planner, runner
from cyclewise.agents.schemas import Approval, CritiqueDraft, Hypothesis, Result, TestPlanDraft
from cyclewise.config import load_config
from cyclewise.eval import report
from cyclewise.record import research_log
from cyclewise.tools import budget, cutoff_view
from cyclewise.tools.cutoff_view import LeakageError
from cyclewise.tools.train_eval import RuleError, score, select_top_k, validate_rule


log = logging.getLogger(__name__)


def _err(e: Exception) -> dict:
    return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def safe(fn):
    """Tools never raise into the agent loop: any failure becomes {"ok": false, "error"}
    so the agent can correct its call. Unexpected errors are also logged."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - deliberate: tool boundary
            if not isinstance(e, (ValueError, KeyError, PermissionError, ValidationError)):
                log.exception("tool %s failed", fn.__name__)
            return _err(e)
    return wrapper


def _revised_rule_before(run_id: str, batch: str) -> dict | None:
    """The rule revised on an EARLIER batch of this run, if any (must be used unchanged)."""
    order = list(load_config()["batches"])
    earlier = set(order[: order.index(batch)]) if batch in order else set()
    rows = [r for r in research_log.rows(run_id=run_id, event="rule_revised") if r["batch"] in earlier]
    return rows[-1]["output"] if rows else None


def _require_batch(batch: str) -> None:
    if batch not in load_config()["batches"]:
        raise ValueError(f"unknown batch {batch!r}; expected one of {list(load_config()['batches'])}")


def _parse(x: Any) -> Any:
    return json.loads(x) if isinstance(x, str) else x


# ---- orchestration ---------------------------------------------------------------

@safe
def start_run() -> dict:
    """Start a CycleWise run. Returns run_id and the batch plan (ids, roles, budget)."""
    cfg = load_config()
    run_id = f"omni-{uuid.uuid4().hex[:8]}"
    research_log.append(run_id, "orchestrator", "run_start", {
        "config_hash": cfg["_hash"], "orchestrator": "omnigent", "llm_backend": "omnigent (claude-sdk harness)",
        "model": "Claude via Omnigent claude-sdk harness", "DEMO_AUTO_APPROVE": False, "budget_cells": budget.total_budget(),
        "batches": {k: b["role"] for k, b in cfg["batches"].items()}})
    return {"ok": True, "run_id": run_id, "budget_cells_per_batch": budget.total_budget(),
            "batches": [{"batch": k, "role": b["role"]} for k, b in cfg["batches"].items()]}


@safe
def get_run_state(run_id: str) -> dict:
    """Latest hypothesis, plan, result, critique and revised rule for a run (no labels of
    unrevealed cells)."""
    rows = research_log.rows(run_id=run_id)
    latest = {}
    for r in rows:
        latest[f"{r['event']}:{r['batch']}"] = r["output"]
    return {"ok": True, "events": [f"{r['seq']}:{r['agent']}:{r['event']}:{r['batch']}" for r in rows],
            "latest": latest}


@safe
def evaluate_run(run_id: str) -> dict:
    """Score the committed selections against both baselines with bootstrap CIs."""
    out = report.evaluate(run_id)
    research_log.append(run_id, "orchestrator", "run_complete", {"log_chain_ok": research_log.verify()})
    return {"ok": True, "markdown": report.to_markdown(out)}


# ---- evidence ----------------------------------------------------------------------

@safe
def list_cells(batch: str) -> dict:
    """Eligible cell ids, charging protocol, and QC status for a batch."""
    _require_batch(batch)
    return {"ok": True, "cells": cutoff_view.list_cells(batch).to_dict("records")}


@safe
def query_early_cycles(batch: str, cell_ids: list[str] | None = None, signals: list[str] | None = None,
                       min_cycle: int = 2, max_cycle: int = 50) -> dict:
    """Per-cycle signals (qd, ir, tmax, chargetime, ...) for cycles min_cycle..max_cycle <= 50."""
    _require_batch(batch)
    try:
        df = cutoff_view.query_early(cell_ids, signals, min_cycle, max_cycle, batch)
        return {"ok": True, "rows": df.head(2000).to_dict("records"), "truncated": len(df) > 2000}
    except (LeakageError, ValueError) as e:
        return _err(e)


@safe
def early_feature_summary(batch: str) -> dict:
    """Feature catalog plus the label-free distribution of each early feature in the batch."""
    _require_batch(batch)
    from cyclewise.data.featurize import FEATURE_CATALOG
    return {"ok": True, "catalog": FEATURE_CATALOG, "summary": cutoff_view.feature_summary(batch)}


@safe
def list_citations() -> dict:
    """The closed list of citation ids agents may use."""
    return {"ok": True, "allowed": sorted(citations.allowed())}


@safe
def submit_hypothesis(run_id: str, batch: str, hypothesis: str) -> dict:
    """Validate and record a Hypothesis JSON {features:[{name,direction,weight}], rationale,
    citations, label:"agent-generated"}. Returns the executable rule derived from it."""
    _require_batch(batch)
    try:
        h = evidence.record(run_id, batch, Hypothesis.model_validate(_parse(hypothesis)), model="omnigent")
        return {"ok": True, "rule": h.as_rule()}
    except (ValidationError, ValueError) as e:
        return _err(e)


# ---- safety ------------------------------------------------------------------------

@safe
def safety_scan(run_id: str, batch: str) -> dict:
    """Flag hot cells, IR jumps and missing sensors in cycles 2..50. Flags go to the human."""
    _require_batch(batch)
    return {"ok": True, "flags": critic_safety.safety_scan(run_id, batch)}


# ---- planner -----------------------------------------------------------------------

@safe
def check_budget(n_cells: int, batch: str) -> dict:
    """Check a request for n_cells against the per-batch cell budget."""
    _require_batch(batch)
    try:
        return budget.check(n_cells, budget.total_budget(), len(cutoff_view.list_cells(batch)))
    except budget.BudgetError as e:
        return _err(e)


@safe
def preview_ranking(batch: str, rule: str, top: int = 15) -> dict:
    """Score cells with an executable rule JSON and return the top of the ranking."""
    _require_batch(batch)
    try:
        r = validate_rule(_parse(rule))
        elig = cutoff_view.list_cells(batch)["cell_id"].tolist()
        s = score(r, batch, elig)
        sel = select_top_k(s, top, elig)
        return {"ok": True, "top": [{"cell_id": c, "score": round(float(s[c]), 4)} for c in sel["cell_ids"]]}
    except (RuleError, ValueError) as e:
        return _err(e)


@safe
def submit_test_plan(run_id: str, batch: str, role: str, plan_draft: str, rule: str,
                     flags: str = "[]") -> dict:
    """Validate a TestPlanDraft JSON {options:[>=2 {option_id, strategy, description, n_cells,
    n_explore, expected_learning}], chosen, reason}, fill cell ids from the rule, record it."""
    try:
        _require_batch(batch)
        rule_d = validate_rule(_parse(rule))
        revised = _revised_rule_before(run_id, batch)
        if revised is not None and rule_d != revised:
            raise ValueError("a revised rule was fit on an earlier batch; it must be used unchanged "
                             f"for {batch} (no tuning on this batch). Pass that rule: {json.dumps(revised)}")
        draft = TestPlanDraft.model_validate(_parse(plan_draft))
        plan = planner.finalize(run_id, batch, draft, rule_d, _parse(flags),
                                budget.total_budget(), model="omnigent")
        return {"ok": True, "plan": plan.model_dump()}
    except (ValidationError, ValueError, RuleError) as e:
        return _err(e)


@safe
def request_approval(run_id: str, plan_id: str, note: str = "") -> dict:
    """Record human approval of a plan. Gated by an Omnigent ASK policy: this tool only
    runs after a human approves the prompt in the Omnigent UI."""
    plan = planner.load_plan(run_id, plan_id)
    a = Approval(plan_id=plan_id, decision="approved", approver="omnigent-ask", note=note)
    research_log.append(run_id, "human", "approval", a.model_dump(), batch=plan.batch, inputs=plan_id)
    return {"ok": True, "approval": a.model_dump()}


# ---- runner ------------------------------------------------------------------------

@safe
def execute_plan(run_id: str, plan_id: str) -> dict:
    """Commit the selection, reveal outcomes of the kept cells (requires approval), log MLflow."""
    plan = planner.load_plan(run_id, plan_id)
    done = research_log.rows(run_id=run_id, event="selection_committed", batch=plan.batch)
    if done:
        raise PermissionError(f"batch {plan.batch} already has an executed plan "
                              f"({done[-1]['output']['plan_id']}); one selection per batch")
    try:
        res = runner.run(run_id, plan)
        return {"ok": True, "result": res.model_dump()}
    except PermissionError as e:
        return _err(e)


# ---- critic ------------------------------------------------------------------------

@safe
def submit_critique(run_id: str, batch: str, role: str, critique: str, revisions_used: int = 0) -> dict:
    """Validate a CritiqueDraft JSON {revise, reason, new_features, flagged_cells, next_experiment}.
    Revision is refused unless the pre-registered trigger fired on the train batch. Role and the
    revision count are read from config and the log; the role/revisions_used arguments are ignored."""
    try:
        _require_batch(batch)
        rows = research_log.rows(run_id=run_id, event="result", batch=batch)
        if not rows:
            raise ValueError(f"no executed result for batch {batch} in run {run_id}; run execute_plan first")
        res_row = rows[-1]
        result = Result(**res_row["output"])
        plan = planner.load_plan(run_id, result.plan_id)
        role = load_config()["batches"][batch]["role"]  # never trust the caller's role
        used = len(research_log.rows(run_id=run_id, event="rule_revised", batch=batch))
        trig, can, max_feats = critic_safety.permitted(result, role, used)
        d = CritiqueDraft.model_validate(_parse(critique))
        if d.revise and not can:
            return {"ok": False, "error": "revision not permitted: trigger did not fire, not the train "
                                          "batch, or revision limit reached", "trigger": trig}
        if d.revise and not 1 <= len(d.new_features) <= max_feats:
            return {"ok": False, "error": f"choose 1..{max_feats} features"}
        crit = critic_safety.finalize(run_id, result, plan.rule, d, trig, model="omnigent")
        if crit.new_rule:
            research_log.append(run_id, "orchestrator", "rule_revised", crit.new_rule, batch=batch)
        return {"ok": True, "critique": crit.model_dump()}
    except (ValidationError, ValueError, IndexError) as e:
        return _err(e)
