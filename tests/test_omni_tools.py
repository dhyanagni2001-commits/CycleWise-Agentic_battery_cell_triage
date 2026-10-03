"""Omnigent tool boundary: tools never raise, and loop rules are enforced from the
log rather than trusted from the caller's arguments."""

import json

import pytest

from cyclewise.agents.planner import fallback_draft
from cyclewise.record import research_log
from cyclewise.tools import omni_tools as T

HYP = {"features": [{"name": "dq_logvar", "direction": -1, "weight": 1.0}],
       "rationale": "Severson 2019 ΔQ variance feature.", "citations": ["severson2019"]}


@pytest.fixture
def run(tmp_log):
    rid = T.start_run()["run_id"]
    rule = T.submit_hypothesis(rid, "b1", json.dumps(HYP))["rule"]
    plan = T.submit_test_plan(rid, "b1", "train", json.dumps(fallback_draft(12, "train").model_dump()),
                              json.dumps(rule))["plan"]
    return rid, rule, plan


@pytest.mark.parametrize("call", [
    lambda: T.request_approval("no-run", "b1-nope"),
    lambda: T.execute_plan("no-run", "b1-nope"),
    lambda: T.submit_critique("no-run", "b1", "train", "{}"),
    lambda: T.submit_hypothesis("r", "b1", "{not json"),
    lambda: T.preview_ranking("b1", '{"kind": "prose"}'),
    lambda: T.list_cells(None),
    lambda: T.submit_test_plan("r", "b9", "train", "{}", "{}"),
])
def test_tools_return_errors_instead_of_raising(call, tmp_log):
    out = call()
    assert out["ok"] is False and out["error"]


def test_one_execution_per_batch(run):
    rid, rule, plan = run
    T.request_approval(rid, plan["plan_id"])
    assert T.execute_plan(rid, plan["plan_id"])["ok"]
    again = T.execute_plan(rid, plan["plan_id"])
    assert again["ok"] is False and "already has an executed plan" in again["error"]


def test_role_and_revision_count_come_from_the_log(run):
    rid, rule, plan = run
    T.request_approval(rid, plan["plan_id"])
    T.execute_plan(rid, plan["plan_id"])
    # Caller claims b1 is "eval" and revisions_used=-5: both ignored. Whether revision is
    # allowed depends only on the pre-registered trigger for the train batch.
    out = T.submit_critique(rid, "b1", "eval", json.dumps({"revise": False, "reason": "checking role spoof"}),
                            revisions_used=-5)
    assert out["ok"] and out["critique"]["trigger"]["thresholds"]


def test_batch2_must_use_revised_rule(run):
    rid, rule, plan = run
    revised = {"kind": "ridge", "features": ["dq_logvar"], "mean": {"dq_logvar": -4.0},
               "std": {"dq_logvar": 0.5}, "coef": {"dq_logvar": -0.2}, "intercept": 2.8,
               "alpha": 1.0, "fit_on": "b1_revealed", "n_train": 12, "target": "log10_cycle_life"}
    research_log.append(rid, "orchestrator", "rule_revised", revised, batch="b1")
    draft = json.dumps(fallback_draft(12, "eval").model_dump())
    other = T.submit_test_plan(rid, "b2", "eval", draft, json.dumps(rule))
    assert other["ok"] is False and "must be used unchanged" in other["error"]
    assert T.submit_test_plan(rid, "b2", "eval", draft, json.dumps(revised))["ok"]
