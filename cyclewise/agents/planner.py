"""Planner agent: proposes >= 2 test options within budget and chooses one.

The LLM chooses strategies and sizes; cell IDs are produced by tools from the
rule, so the model cannot hallucinate a cell. Any ID that does reach a plan is
validated against the eligible set.
"""

from __future__ import annotations

import json
import uuid

from cyclewise.agents import llm
from cyclewise.agents.schemas import TestOption, TestPlan, TestPlanDraft
from cyclewise.config import load_config
from cyclewise.record import research_log
from cyclewise.tools import budget
from cyclewise.tools.cutoff_view import list_cells
from cyclewise.tools.train_eval import score, select_top_k, stratified_explore

SYSTEM = """You are the Planner agent in First Fifty. Each test option keeps a set of cells on test
past cycle 50 until end of life; the budget is counted in cells. Propose at least two distinct
options and choose one. Strategies:
- exploit_top_k: keep the K highest-scoring cells (maximises long-lived cells found now).
- explore_stratified: keep K - n_explore top cells plus n_explore cells spread across the score
  range (finds slightly fewer long-lived cells now, but the revealed outcomes span the score axis,
  which makes a revised rule fit on them more reliable).
- hedge_with_dq: half from the hypothesis rule, half from the published ΔQ-variance ranking.
Use exploration only when a later batch can benefit from what is learned (role=train).
Never request more cells than the remaining budget."""


def fallback_draft(k: int, role: str) -> TestPlanDraft:
    n_exp = max(1, k // 3)
    opts = [
        TestOption(option_id="A", strategy="exploit_top_k", n_cells=k, description=f"top {k} by rule score",
                   expected_learning="low: outcomes only for top-ranked cells"),
        TestOption(option_id="B", strategy="explore_stratified", n_cells=k, n_explore=n_exp,
                   description=f"top {k - n_exp} by score + {n_exp} spread over the score range",
                   expected_learning="high: outcomes span the score axis for re-fitting the rule"),
        TestOption(option_id="C", strategy="hedge_with_dq", n_cells=k,
                   description="half rule, half ΔQ-variance ranking",
                   expected_learning="medium: compares rule against the published feature"),
    ]
    chosen = "B" if role == "train" else "A"
    return TestPlanDraft(options=opts, chosen=chosen,
                         reason=("train batch: a revision can still be applied downstream, so buy "
                                 "learning" if role == "train" else
                                 "eval batch: no later batch benefits from exploration; exploit"))


def _materialize(opt: TestOption, rule: dict, batch: str, eligible: list[str]) -> dict:
    s = score(rule, batch, eligible)
    if opt.strategy == "exploit_top_k":
        sel = select_top_k(s, opt.n_cells, eligible)
    elif opt.strategy == "explore_stratified":
        sel = stratified_explore(s, opt.n_cells, min(opt.n_explore, opt.n_cells - 1), eligible)
    else:
        dq = score({"kind": "zscore_sum", "weights": {"dq_logvar": -1.0}}, batch, eligible)
        half = opt.n_cells // 2
        a = select_top_k(s, half, eligible)["cell_ids"]
        rest = [c for c in select_top_k(dq, len(eligible), eligible)["ranking"] if c not in a]
        sel = {"cell_ids": a + rest[: opt.n_cells - half], "tie_at_boundary": False}
    return sel


def run(run_id: str, batch: str, role: str, rule: dict, flags: list[dict],
        remaining: int, note: str = "") -> TestPlan:
    eligible = list_cells(batch)["cell_id"].tolist()
    k = min(remaining, budget.total_budget())

    def check(d: TestPlanDraft) -> None:
        for o in d.options:
            budget.check(o.n_cells, remaining, len(eligible))
            if o.n_explore >= o.n_cells:
                raise ValueError(f"option {o.option_id}: n_explore must be < n_cells")

    prompt = (f"Batch {batch} (role={role}). Eligible cells: {len(eligible)}. Remaining budget: "
              f"{remaining} cells.\nRule in force: {json.dumps(rule)}\n"
              f"Safety flags (go to the human reviewer): {json.dumps(flags)}\n")
    if note:
        prompt += f"\nThe human reviewer REJECTED the previous plan with this note: {note!r}. Re-propose.\n"
    out = llm.complete(TestPlanDraft, SYSTEM, prompt, fallback=lambda: fallback_draft(k, role), check=check)
    return finalize(run_id, batch, out.value, rule, flags, remaining, note=note, model=out.model,
                    fallback_used=out.fallback_used, llm_errors=out.errors)


def finalize(run_id: str, batch: str, draft: TestPlanDraft, rule: dict, flags: list[dict],
             remaining: int, note: str = "", model: str = "", fallback_used: bool = False,
             llm_errors: list[str] | None = None) -> TestPlan:
    """Turn a validated draft into a concrete plan (cell ids from tools) and log it."""
    cfg = load_config()
    eligible = list_cells(batch)["cell_id"].tolist()
    for o in draft.options:
        budget.check(o.n_cells, remaining, len(eligible))
    chosen = next(o for o in draft.options if o.option_id == draft.chosen)
    sel = _materialize(chosen, rule, batch, eligible)
    unknown = [c for c in sel["cell_ids"] if c not in eligible]
    if unknown:
        raise ValueError(f"plan contains unknown cell ids {unknown}")
    budget_status = budget.check(len(sel["cell_ids"]), remaining, len(eligible))
    plan = TestPlan(
        plan_id=f"{batch}-{uuid.uuid4().hex[:8]}", batch=batch, options=draft.options,
        chosen=draft.chosen, reason=draft.reason, cost_cells=len(sel["cell_ids"]),
        est_cost_cycles=chosen.est_cost_cycles, expected_learning=chosen.expected_learning,
        cell_ids=sel["cell_ids"], rule=rule, flagged_cells=flags,
        tie_at_boundary=sel.get("tie_at_boundary", False),
    )
    research_log.append(run_id, "planner", "test_plan",
                        {**plan.model_dump(), "budget": budget_status, "llm_errors": llm_errors or []},
                        batch=batch, inputs={"rule": rule, "remaining": remaining, "note": note},
                        model=model, prompt_version=cfg["agents"]["prompt_version"],
                        fallback_used=fallback_used)
    return plan


def load_plan(run_id: str, plan_id: str) -> TestPlan:
    rows = [r for r in research_log.rows(run_id=run_id, event="test_plan") if r["output"]["plan_id"] == plan_id]
    if not rows:
        raise KeyError(f"unknown plan {plan_id} in run {run_id}")
    o = {k: v for k, v in rows[-1]["output"].items() if k not in ("budget", "llm_errors")}
    return TestPlan(**o)
