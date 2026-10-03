"""Budget, schema, citation, rule, log, and label edge cases (spec 4.2, 4.4, 4.5)."""

import copy

import pandas as pd
import pytest
from pydantic import ValidationError

from cyclewise.agents import citations
from cyclewise.agents.planner import fallback_draft
from cyclewise.agents.schemas import Hypothesis
from cyclewise.agents.schemas import TestPlanDraft as PlanDraft
from cyclewise.config import ConfigError, _validate, load_config
from cyclewise.data.load_raw import compute_label
from cyclewise.data.splits import label_long_lived
from cyclewise.policies import omni_policies
from cyclewise.record import research_log
from cyclewise.tools import budget
from cyclewise.tools.train_eval import RuleError, select_top_k, validate_rule


@pytest.mark.parametrize("b", [0, -3, 2.5, True])
def test_bad_budget_rejected_at_config_load(b):
    cfg = copy.deepcopy(load_config())
    cfg["preregistered"]["budget_cells"] = b
    with pytest.raises(ConfigError):
        _validate(cfg)


def test_budget_check():
    with pytest.raises(budget.BudgetError):
        budget.check(13, 12, 40)
    assert "trivially" in budget.check(12, 50, 40)["warning"]


def test_budget_policy_denies_oversize_plan():
    d = fallback_draft(12, "train").model_dump()
    d["options"][0]["n_cells"] = 99
    ev = {"type": "tool_call", "data": {"name": "submit_test_plan", "arguments": {"plan_draft": d}}}
    assert omni_policies.budget_guard(ev)["result"] == "DENY"


def test_planner_needs_two_options():
    d = fallback_draft(12, "train").model_dump()
    d["options"] = d["options"][:1]
    with pytest.raises(ValidationError):
        PlanDraft.model_validate(d)
    d = fallback_draft(12, "train").model_dump()
    d["chosen"] = "Z"
    with pytest.raises(ValidationError):
        PlanDraft.model_validate(d)


def test_citations_closed_list():
    assert "severson2019" in citations.allowed()
    with pytest.raises(ValueError):
        citations.validate(["smith2021_made_up"])


def test_hypothesis_rejects_unknown_feature():
    with pytest.raises(ValidationError):
        Hypothesis(features=[{"name": "cycle_life", "direction": 1}], rationale="x" * 30,
                   citations=["severson2019"])


def test_rule_must_be_executable():
    with pytest.raises(RuleError):
        validate_rule({"kind": "prose", "text": "pick good cells"})
    with pytest.raises(RuleError):
        validate_rule({"kind": "zscore_sum", "weights": {"not_a_feature": 1}})


def test_tie_break_by_cell_id():
    s = pd.Series({"b1c9": 1.0, "b1c2": 1.0, "b1c3": 0.5})
    out = select_top_k(s, 1)
    assert out["cell_ids"] == ["b1c2"] and out["tie_at_boundary"]


def test_log_chain_detects_tampering(tmp_log):
    research_log.append("r", "a", "e", {"x": 1})
    research_log.append("r", "a", "e", {"x": 2})
    assert research_log.verify()
    with research_log._db() as con:
        con.execute("UPDATE research_log SET output_json = '{\"x\": 99}' WHERE seq = 1")
    assert not research_log.verify()


def test_censored_and_stopped_labels():
    df = pd.DataFrame({"cycle": [1, 2, 3], "qd": [1.0, 0.95, 0.882], "glitch": False})
    assert compute_label(df, 0.88, 0.005) == {"cycle_life": 4, "censored": False, "last_cycle": 3,
                                              "label_source": "stopped_at_eol"}
    df["qd"] = [1.0, 0.99, 0.97]
    assert compute_label(df, 0.88, 0.005)["censored"]
    lab = pd.DataFrame({"cycle_life": [500, 900, 900], "censored": [True, True, False]})
    out = label_long_lived(lab, 900)
    assert list(out["label_known"]) == [False, True, True]
    assert out["long_lived"].tolist()[1:] == [True, True]  # tie at threshold counts (>=)
