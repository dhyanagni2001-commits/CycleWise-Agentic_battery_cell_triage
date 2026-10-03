"""v2: pre-registration guard, paid-for-data leakage control, budgets, options."""

import pytest

from cyclewise import prereg
from cyclewise.agents.schemas import Approval
from cyclewise.record import research_log
from cyclewise.tools.cutoff_view import LeakageError
from cyclewise.v2 import visible
from cyclewise.v2.allocation import Budget, cut_to_budget, options_at_50, options_at_100, options_at_150


def test_prereg_is_committed_and_hash_locked():
    meta = prereg.verify()
    assert meta["prereg_commit"] and len(meta["prereg_sha256"]) == 64


def test_prereg_tamper_detected(monkeypatch, tmp_path):
    fake = tmp_path / "prereg_v2.yaml"
    fake.write_text(prereg.PREREG.read_text() + "\n# edited after commit\n")
    monkeypatch.setattr(prereg, "PREREG", fake)
    with pytest.raises(prereg.PreregError, match="differs from its frozen hash"):
        prereg.verify()


@pytest.mark.parametrize("n,k,ext", [(46, 12, 1750), (43, 11, 1650), (40, 10, 1500), (1, 1, 50)])
def test_budget_formulas(n, k, ext):
    b = Budget.for_batch(n)
    assert (b.k_final, b.extension) == (k, ext)


def test_options_are_affordable_and_at_least_two():
    b = Budget.for_batch(46)
    for spent in (0, 600, 1500):
        for opts in (options_at_50(b, 46, spent), options_at_100(b, 23, spent)):
            assert len(opts) >= 2 and len({o["option_id"] for o in opts}) == len(opts)
            assert all(o["m"] * 50 <= b.extension - spent for o in opts)
    o50 = options_at_50(b, 46, 0)
    assert all(o["m"] * 50 + b.k_final * 50 <= b.extension for o in o50)  # reserves the 100->150 step
    assert len(options_at_150(b, 20)) >= 2


def test_cut_to_budget_by_rank():
    cells, cut = cut_to_budget(["b1c3", "b1c1", "b1c2"], 3, 100)
    assert cells == ["b1c3", "b1c1"] and cut


@pytest.fixture
def run(tmp_log):
    rid = f"t-{tmp_log.name[-8:].replace('_', '')}"
    visible.create(rid)
    yield rid
    visible.db_path(rid).unlink(missing_ok=True)
    visible.db_path(rid).with_name(visible.db_path(rid).name + ".lock").unlink(missing_ok=True)


@pytest.mark.requires_data
def test_unpaid_cycles_are_unreachable(run):
    with pytest.raises(LeakageError, match="not paid for"):
        visible.query_cycles(run, ["b1c3"], ["qd"], 51)
    with pytest.raises(LeakageError, match="not paid for"):
        visible.qdlin_at(run, ["b1c3"], 100)
    assert visible.query_cycles(run, ["b1c3"], ["qd"], 50)["cycle"].max() == 50


@pytest.mark.requires_data
def test_materialize_requires_commit_and_approval_and_pays_only_those_cells(run):
    with pytest.raises(PermissionError, match="committed and approved"):
        visible.materialize(run, "b1", ["b1c3"], 50, 100, "p1")
    research_log.append(run, "runner", "checkpoint_committed", {"plan_id": "p1", "extend": ["b1c3"], "keep": []})
    with pytest.raises(PermissionError, match="committed and approved"):
        visible.materialize(run, "b1", ["b1c3"], 50, 100, "p1")
    research_log.append(run, "human", "approval", Approval(plan_id="p1", decision="approved", approver="t").model_dump())
    with pytest.raises(PermissionError, match="differ"):
        visible.materialize(run, "b1", ["b1c3", "b1c4"], 50, 100, "p1")
    visible.materialize(run, "b1", ["b1c3"], 50, 100, "p1")
    assert visible.query_cycles(run, ["b1c3"], ["qd"], 100)["cycle"].max() == 100
    assert len(visible.qdlin_at(run, ["b1c3"], 100)) == 1
    with pytest.raises(LeakageError):
        visible.query_cycles(run, ["b1c3"], ["qd"], 101)        # paid to 100, not 101
    with pytest.raises(LeakageError):
        visible.query_cycles(run, ["b1c4"], ["qd"], 100)        # b1c4 was not extended


@pytest.mark.requires_data
def test_visible_db_has_no_labels_and_nothing_past_paid(run):
    with visible.locked(visible.db_path(run)) as con:
        tables = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
        assert not tables & {"labels", "cycles_raw", "qdlin_ckpt"}
        assert con.execute("SELECT max(cycle) FROM cycles").fetchone()[0] <= 50
        assert con.execute("SELECT max(cycle) FROM qdlin").fetchone()[0] <= 50
