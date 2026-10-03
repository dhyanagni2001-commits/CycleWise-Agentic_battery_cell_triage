"""Leakage tests (spec 4.3). Shown in the demo: cycle 51 is unreachable through
every agent-facing path, and labels are unreachable before approval."""

import duckdb
import pytest

from cyclewise.agents.schemas import Approval
from cyclewise.config import load_config, path
from cyclewise.policies import omni_policies
from cyclewise.record import research_log
from cyclewise.tools import cutoff_view, omni_tools
from cyclewise.tools.cutoff_view import LeakageError
from cyclewise.tools.reveal import ApprovalError, commit_selection, reveal


def tool_event(name, **arguments):
    return {"type": "tool_call", "data": {"name": name, "arguments": arguments}}


# ---- layer 1: physical separation -------------------------------------------------

def test_early_db_has_no_row_past_50_and_no_labels():
    with duckdb.connect(str(path(load_config()["storage"]["early_db"])), read_only=True) as con:
        assert con.execute("SELECT max(cycle) FROM cycles_early_t").fetchone()[0] <= 50
        assert con.execute("SELECT max(cycle) FROM qdlin_early").fetchone()[0] <= 50
        tables = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
        cols = {r[0] for r in con.execute("SELECT column_name FROM information_schema.columns").fetchall()}
    assert not tables & {"labels_hidden", "cycles_raw", "frozen_params"}
    assert not cols & {"cycle_life", "long_lived", "censored", "file_cycle_life"}


def test_agent_connection_is_read_only():
    with cutoff_view._con() as con:
        with pytest.raises(Exception):
            con.execute("CREATE TABLE x AS SELECT 1")


# ---- layer 2: tool code -----------------------------------------------------------

@pytest.mark.parametrize("call", [
    lambda: cutoff_view.query_early(max_cycle=51),
    lambda: cutoff_view.query_early(min_cycle=51),
    lambda: cutoff_view.query_early(max_cycle=1000, batch="b1"),
    lambda: cutoff_view.check_cycle(51),
    lambda: cutoff_view.check_cycle(50.5),
])
def test_cycle_51_rejected_by_tool_code(call):
    with pytest.raises(LeakageError):
        call()


def test_cycle_51_rejected_by_omnigent_tool_wrapper():
    out = omni_tools.query_early_cycles("b1", max_cycle=51)
    assert out["ok"] is False and "LeakageError" in out["error"]
    out = omni_tools.query_early_cycles("b1", min_cycle=51, max_cycle=50)
    assert out["ok"] is False


def test_cycle_50_allowed():
    df = cutoff_view.query_early(batch="b1", max_cycle=50)
    assert df["cycle"].max() == 50


def test_features_have_no_label_columns():
    df = cutoff_view.get_features("b2")
    assert not {"cycle_life", "long_lived", "censored"} & set(df.columns)


# ---- layer 3: Omnigent policy ------------------------------------------------------

@pytest.mark.parametrize("name,args", [
    ("query_early_cycles", {"batch": "b1", "max_cycle": 51}),
    ("query_early_cycles", {"batch": "b1", "min_cycle": 2, "max_cycle": 500}),
    ("anything", {"nested": {"cycle": 51}}),
    ("sys_os_shell", {"command": "duckdb warehouse/hidden.duckdb 'select * from labels_hidden'"}),
    ("anything", {"sql": "select * from cycles_raw"}),
])
def test_cycle_51_and_hidden_tables_denied_by_policy(name, args):
    assert omni_policies.leakage_guard(tool_event(name, **args))["result"] == "DENY"


def test_policy_allows_cycle_50():
    ev = tool_event("query_early_cycles", batch="b1", max_cycle=50)
    assert omni_policies.leakage_guard(ev)["result"] == "ALLOW"


# ---- labels unreachable before approval ----------------------------------------------

def test_reveal_requires_commit_and_approval(tmp_log):
    run, cells = "t-run", ["b1c5", "b1c6"]
    with pytest.raises(ApprovalError, match="not committed"):
        reveal(run, "b1", "p1", cells)
    commit_selection(run, "b1", "p1", cells)
    with pytest.raises(ApprovalError, match="no human approval"):
        reveal(run, "b1", "p1", cells)
    research_log.append(run, "human", "approval",
                        Approval(plan_id="p1", decision="rejected", approver="t").model_dump())
    with pytest.raises(ApprovalError, match="no human approval"):
        reveal(run, "b1", "p1", cells)
    research_log.append(run, "human", "approval",
                        Approval(plan_id="p1", decision="approved", approver="t").model_dump())
    with pytest.raises(ApprovalError, match="differ"):
        reveal(run, "b1", "p1", ["b1c5", "b1c7"])
    out = reveal(run, "b1", "p1", cells)
    assert sorted(out["cell_id"]) == cells  # only the selected cells come back


def test_reveal_gate_policy_denies_without_approval(tmp_log):
    ev = tool_event("execute_plan", run_id="nope", plan_id="b2-xxxx")
    assert omni_policies.reveal_gate(ev)["result"] == "DENY"


def test_approval_gate_always_asks():
    ev = tool_event("request_approval", run_id="r", plan_id="p")
    assert omni_policies.approval_gate(ev)["result"] == "ASK"
