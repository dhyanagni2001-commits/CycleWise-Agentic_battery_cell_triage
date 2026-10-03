"""Runner agent: executes an approved plan. Owns no decision.

Order matters: commit selection (ids + hash) -> reveal (gated on commit + approval)
-> log to MLflow. Metrics here use revealed cells only (what the lab would know).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from cyclewise.agents.schemas import Result, TestPlan
from cyclewise.eval import tracking
from cyclewise.record import research_log
from cyclewise.tools.reveal import commit_selection, reveal
from cyclewise.tools.train_eval import score


def revealed_metrics(plan: TestPlan, outcomes) -> dict:
    known = outcomes[outcomes["label_known"]]
    hits = int(known["long_lived"].astype(bool).sum())
    s = score(plan.rule, plan.batch, outcomes["cell_id"].tolist())
    life = np.log10(outcomes.set_index("cell_id")["cycle_life"].astype(float))
    rho = float(spearmanr(s.loc[life.index], life).statistic) if len(life) > 2 else float("nan")
    return {"n_selected": len(outcomes), "n_label_known": int(len(known)), "hits": hits,
            "precision_at_k": hits / max(1, len(known)), "spearman_revealed": rho,
            "realized_test_cycles": int(outcomes["cycle_life"].sum())}


def run(run_id: str, plan: TestPlan) -> Result:
    commit_selection(run_id, plan.batch, plan.plan_id, plan.cell_ids)
    outcomes = reveal(run_id, plan.batch, plan.plan_id, plan.cell_ids)
    m = revealed_metrics(plan, outcomes)
    mlflow_id = tracking.log_selection(
        name=f"cyclewise_{plan.batch}", batch=plan.batch, method="cyclewise",
        params={"plan_id": plan.plan_id, "strategy": plan.chosen, "rule_kind": plan.rule["kind"],
                "cost_cells": plan.cost_cells, "run_id": run_id},
        metrics={k: v for k, v in m.items() if isinstance(v, (int, float))},
        selected=plan.cell_ids, rule=plan.rule)
    res = Result(run_id=run_id, plan_id=plan.plan_id, batch=plan.batch, mlflow_run_id=mlflow_id,
                 selected_ids=plan.cell_ids, revealed_outcomes=outcomes.to_dict("records"), metrics=m)
    research_log.append(run_id, "runner", "result", res.model_dump(), batch=plan.batch,
                        inputs=plan.plan_id)
    return res
