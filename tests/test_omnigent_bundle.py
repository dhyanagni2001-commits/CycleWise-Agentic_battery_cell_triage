"""The Omnigent bundle must load with Omnigent's own parser and expose every tool."""

from pathlib import Path

import pytest

omnigent = pytest.importorskip("omnigent")
from omnigent.spec import load  # noqa: E402
from omnigent.tools.local import load_local_python_tools  # noqa: E402

from omnigent_gen import AGENT_TOOLS  # noqa: E402

BUNDLE = Path(__file__).resolve().parent.parent / "omnigent" / "cyclewise_lab"


def test_bundle_loads_with_four_subagents_and_policies():
    spec = load(BUNDLE)
    subs = {a.name: a for a in spec.sub_agents}
    assert set(subs) == {"evidence", "planner", "runner", "critic_safety"}
    for a in [spec, *spec.sub_agents]:
        names = [p.name for p in a.guardrails.policies]
        assert "leakage_guard" in names
    # The human gate must sit where the human is: the supervisor session.
    assert "approval_gate" in [p.name for p in spec.guardrails.policies]
    assert "approval_gate" not in [p.name for p in subs["planner"].guardrails.policies]
    assert "reveal_gate" in [p.name for p in subs["runner"].guardrails.policies]


def test_every_agent_exposes_its_tools():
    spec = load(BUNDLE)
    pairs = [(".", spec)] + [(f"agents/{a.source_rel_dir}", a) for a in spec.sub_agents]
    for rel, a in pairs:
        tools = load_local_python_tools(a.local_tools, BUNDLE / rel, sandbox_enabled=False)
        assert sorted(t.name() for t in tools) == sorted(AGENT_TOOLS[rel])


def test_defaulted_arguments_stay_optional():
    """Strict tool schemas make every argument required; a model omitting note="" was
    rejected live. Tools are generated non-strict so defaults stay optional."""
    import importlib.util
    from omnigent_client.tools import get_tool_metadata
    f = BUNDLE / "tools" / "python" / "request_approval.py"
    spec = importlib.util.spec_from_file_location("ra", f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert get_tool_metadata(m.request_approval).json_schema["required"] == ["run_id", "plan_id"]


def test_every_tool_is_granted_at_dispatch():
    """The check that refused `start_run` in a live `omni run`: every tool must be in the
    session's granted surface, for the harness the bundle declares."""
    from omnigent.runner.tool_dispatch import _ungranted_tool_reason
    spec = load(BUNDLE)
    pairs = [(".", spec)] + [(f"agents/{a.source_rel_dir}", a) for a in spec.sub_agents]
    for rel, a in pairs:
        for name in AGENT_TOOLS[rel]:
            assert _ungranted_tool_reason(name, a, "claude-sdk") is None, (a.name, name)
        assert _ungranted_tool_reason("not_a_tool", a, "claude-sdk") is not None
