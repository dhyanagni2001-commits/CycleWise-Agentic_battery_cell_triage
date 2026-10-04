"""v2 Omnigent bundle (sequential checkpoints): loads with Omnigent's parser, every tool is
granted at dispatch, gates are on the right agents, and the tool flow enforces order and approval."""

import json
from pathlib import Path

import pytest

pytest.importorskip("omnigent")
from omnigent.runner.tool_dispatch import _ungranted_tool_reason  # noqa: E402
from omnigent.spec import load  # noqa: E402
from omnigent.tools.local import load_local_python_tools  # noqa: E402

from omnigent_gen import AGENT_TOOLS_V2  # noqa: E402

BUNDLE = Path(__file__).resolve().parent.parent / "omnigent" / "cyclewise_v2"


def _pairs():
    spec = load(BUNDLE)
    return spec, [(".", spec)] + [(f"agents/{a.source_rel_dir}", a) for a in spec.sub_agents]


def test_v2_bundle_structure_and_gates():
    spec, _ = _pairs()
    subs = {a.name: a for a in spec.sub_agents}
    assert set(subs) == {"evidence", "planner", "runner", "critic_safety"}
    pol = lambda a: [p.name for p in a.guardrails.policies]  # noqa: E731
    assert all("leakage_guard" in pol(a) for a in [spec, *spec.sub_agents])
    assert "approval_gate" in pol(spec) and "approval_gate" not in pol(subs["planner"])
    assert "reveal_gate" in pol(subs["runner"])


def test_v2_every_tool_loads_and_is_granted():
    _, pairs = _pairs()
    for rel, a in pairs:
        tools = load_local_python_tools(a.local_tools, BUNDLE / rel, sandbox_enabled=False)
        assert sorted(t.name() for t in tools) == sorted(AGENT_TOOLS_V2[rel])
        for name in AGENT_TOOLS_V2[rel]:
            assert _ungranted_tool_reason(name, a, "claude-sdk") is None, (a.name, name)


@pytest.mark.requires_data
def test_v2_tool_flow_enforces_order_and_approval(tmp_log, monkeypatch):
    from cyclewise.policies import omni_policies as P
    from cyclewise.tools import omni_tools_v2 as T
    from cyclewise.v2 import visible
    r = T.v2_start_run()
    assert r["ok"], r
    rid = r["run_id"]
    try:
        hyp = json.dumps({"features": [{"name": "dq_logvar", "direction": -1, "weight": 1.0}],
                          "rationale": "Severson 2019 ΔQ variance base signal.", "citations": ["severson2019"]})
        assert not T.v2_submit_hypothesis(rid, "b2", hyp)["ok"]            # b1 first
        assert not T.v2_checkpoint_options(rid, "b1")["ok"]               # needs hypothesis + scan
        assert T.v2_submit_hypothesis(rid, "b1", hyp)["ok"] and T.v2_safety_scan(rid, "b1")["ok"]
        o = T.v2_checkpoint_options(rid, "b1")
        assert o["ok"] and o["checkpoint"] == 50 and len(o["options"]) >= 2
        assert not T.v2_submit_choice(rid, "b1", "Z", "not an option at all")["ok"]
        pl = T.v2_submit_choice(rid, "b1", o["options"][-1]["option_id"], "narrow-deep keeps a margin at 150")
        assert pl["ok"]
        exe = {"type": "tool_call", "data": {"name": "v2_execute_checkpoint",
                                             "arguments": {"run_id": rid, "plan_id": pl["plan_id"]}}}
        assert P.reveal_gate(exe)["result"] == "DENY"
        assert not T.v2_execute_checkpoint(rid, pl["plan_id"])["ok"]      # no approval yet
        ask = {"type": "tool_call", "data": {"name": "v2_request_approval",
                                             "arguments": {"run_id": rid, "plan_id": pl["plan_id"]}}}
        assert P.approval_gate(ask)["result"] == "ASK"
        assert T.v2_request_approval(rid, pl["plan_id"])["ok"]
        assert P.reveal_gate(exe)["result"] == "ALLOW"
        assert T.v2_execute_checkpoint(rid, pl["plan_id"])["ok"]
        assert not T.v2_execute_checkpoint(rid, pl["plan_id"])["ok"]      # once only
        assert T.v2_checkpoint_options(rid, "b1")["checkpoint"] == 100
        assert not T.v2_submit_critique(rid, "b1", json.dumps({"revise": False, "reason": "too early to critique"}))["ok"]
    finally:
        p = visible.db_path(rid)
        p.unlink(missing_ok=True)
        p.with_name(p.name + ".lock").unlink(missing_ok=True)
