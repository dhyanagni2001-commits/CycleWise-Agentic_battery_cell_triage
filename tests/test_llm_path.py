"""The LLM path, exercised with a scripted fake model (no network, no key).

Covers: JSON in a code fence, retry after a validation error, fallback after
repeated garbage, fallback on a network error, domain checks (citations,
budget, revision permission), and a full two-batch loop that takes the
revision branch.
"""

import copy
import json

import pytest

from cyclewise.agents import llm
from cyclewise.agents.planner import fallback_draft
from cyclewise.agents.schemas import CritiqueDraft, Hypothesis, TestPlanDraft
from cyclewise.config import load_config

GOOD_HYP = {"features": [{"name": "dq_logvar", "direction": -1, "weight": 1.0},
                         {"name": "fade_slope_late", "direction": 1, "weight": 0.5}],
            "rationale": "Severson 2019 ΔQ variance plus late fade slope.", "citations": ["severson2019"],
            "label": "agent-generated"}


@pytest.fixture
def fake_llm(monkeypatch, tmp_path):
    """Install a scripted model: responses[schema_title] is a list consumed in order;
    the last entry repeats. An Exception instance is raised instead of returned."""
    monkeypatch.setenv("CYCLEWISE_LLM", "anthropic")
    monkeypatch.setattr(llm, "CACHE", tmp_path / "llm_cache")
    monkeypatch.setattr(llm._Budget, "calls", 0)
    script: dict[str, list] = {}
    calls: list[str] = []

    def fake(system: str, prompt: str) -> str:
        title = json.loads(system.split("nothing else:\n", 1)[1])["title"]
        calls.append(title)
        seq = script[title]
        item = seq.pop(0) if len(seq) > 1 else seq[0]
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, str) else json.dumps(item)

    monkeypatch.setattr(llm, "_call_anthropic", fake)
    return script, calls


def test_fenced_json_accepted(fake_llm):
    script, _ = fake_llm
    script["Hypothesis"] = [f"Here you go:\n```json\n{json.dumps(GOOD_HYP)}\n```"]
    out = llm.complete(Hypothesis, "s", "p", fallback=lambda: pytest.fail("no fallback expected"))
    assert not out.fallback_used and out.value.features[1].name == "fade_slope_late"


def test_retry_after_invalid_then_valid(fake_llm):
    script, calls = fake_llm
    bad = dict(GOOD_HYP, features=[{"name": "made_up_feature", "direction": 1}])
    script["Hypothesis"] = [bad, GOOD_HYP]
    out = llm.complete(Hypothesis, "s", "p", fallback=lambda: pytest.fail("no fallback expected"))
    assert out.attempts == 2 and not out.fallback_used and "made_up_feature" in out.errors[0]


def test_fallback_after_repeated_garbage(fake_llm):
    script, calls = fake_llm
    script["Hypothesis"] = ["no json here at all"]
    from cyclewise.agents.evidence import fallback_hypothesis
    out = llm.complete(Hypothesis, "s", "p", fallback=fallback_hypothesis)
    assert out.fallback_used and len(calls) == load_config()["agents"]["max_json_retries"] + 1


def test_fallback_on_network_error_no_retry_loop(fake_llm):
    script, calls = fake_llm
    script["Hypothesis"] = [ConnectionError("network down")]
    from cyclewise.agents.evidence import fallback_hypothesis
    out = llm.complete(Hypothesis, "s", "p", fallback=fallback_hypothesis)
    assert out.fallback_used and "network down" in out.errors[-1]


def test_hallucinated_citation_rejected(fake_llm):
    script, _ = fake_llm
    script["Hypothesis"] = [dict(GOOD_HYP, citations=["nobody2099"]), GOOD_HYP]
    from cyclewise.agents import citations
    out = llm.complete(Hypothesis, "s", "p", fallback=lambda: pytest.fail("x"),
                       check=lambda h: citations.validate(h.citations))
    assert out.value.citations == ["severson2019"] and "nobody2099" in out.errors[0]


def test_cache_replays_without_calls(fake_llm):
    script, calls = fake_llm
    script["Hypothesis"] = [GOOD_HYP]
    llm.complete(Hypothesis, "s", "same prompt", fallback=lambda: pytest.fail("x"))
    n = len(calls)
    out = llm.complete(Hypothesis, "s", "same prompt", fallback=lambda: pytest.fail("x"))
    assert len(calls) == n and out.cached


@pytest.mark.requires_data
def test_full_loop_with_llm_and_revision(fake_llm, tmp_log, tmp_path, monkeypatch):
    """Two batches through the real loop with the fake model. The trigger thresholds are
    raised IN THIS TEST ONLY so the revision branch runs end to end."""
    from cyclewise.eval import report
    from cyclewise.record import research_log
    from cyclewise import run_loop

    monkeypatch.setenv("DEMO_AUTO_APPROVE", "true")
    monkeypatch.setattr(report, "REPORTS", tmp_path / "reports")
    cfg = load_config()
    trig = copy.deepcopy(cfg["preregistered"]["revision_trigger"])
    monkeypatch.setitem(cfg["preregistered"], "revision_trigger",
                        dict(trig, min_precision_at_k=1.01))  # always fires

    plan = fallback_draft(12, "train").model_dump()
    over = copy.deepcopy(plan)
    over["options"][0]["n_cells"] = 99                               # over budget -> retried
    script, calls = fake_llm
    script["Hypothesis"] = [GOOD_HYP]
    script["TestPlanDraft"] = [over, plan]
    script["CritiqueDraft"] = [
        {"revise": True, "reason": "trigger fired; refit on revealed cells", "new_features": ["dq_logvar", "q2"],
         "next_experiment": "sequential checkpoints"},
        {"revise": True, "reason": "batch 2: should be refused", "new_features": ["dq_logvar"]},
        {"revise": False, "reason": "eval batch; no revision permitted", "next_experiment": "prospective"},
    ]
    run_id = run_loop.run("llm-test")
    rows = research_log.rows(run_id=run_id)
    events = [r["event"] for r in rows]
    assert "rule_revised" in events
    plans = [r["output"] for r in rows if r["event"] == "test_plan"]
    revised = next(r["output"] for r in rows if r["event"] == "rule_revised")
    assert revised["fit_on"] == "b1_revealed" and revised["features"] == ["dq_logvar", "q2"]
    assert plans[1]["rule"] == revised                       # batch 2 uses the b1 rule unchanged
    crit_b2 = [r["output"] for r in rows if r["event"] == "critique" and r["batch"] == "b2"][-1]
    assert crit_b2["revise"] is False                        # refused on the eval batch
    assert not any(r["fallback_used"] for r in rows if r["event"] in ("hypothesis", "test_plan"))
    assert research_log.verify()
    assert (tmp_path / "reports" / f"{run_id}.md").exists()


def test_schemas_are_json_schema_exportable():
    for s in (Hypothesis, TestPlanDraft, CritiqueDraft):
        assert json.loads(json.dumps(s.model_json_schema()))["title"] == s.__name__
