"""reveal(): the only path from agents to labels.

Opens the hidden DB only if the research log holds, for this exact plan_id:
  1. a `selection_committed` row whose selection hash matches the requested cells, and
  2. an `approval` row with decision == "approved" written after that commit.
Returns outcomes for the selected cells only (unselected cells were "stopped at
cycle 50" and stay unknown).
"""

from __future__ import annotations

import pandas as pd

from cyclewise.config import load_config, path
from cyclewise.record.dblock import connect_ro
from cyclewise.record import research_log
from cyclewise.record.research_log import stable_hash


class ApprovalError(PermissionError):
    pass


def selection_hash(cell_ids: list[str]) -> str:
    return stable_hash(sorted(cell_ids))[:16]


def commit_selection(run_id: str, batch: str, plan_id: str, cell_ids: list[str]) -> dict:
    out = {"plan_id": plan_id, "cell_ids": sorted(cell_ids), "selection_hash": selection_hash(cell_ids)}
    research_log.append(run_id, "runner", "selection_committed", out, batch=batch, inputs=cell_ids)
    return out


def _authorized(run_id: str, plan_id: str, cell_ids: list[str]) -> None:
    log = research_log.rows(run_id=run_id)
    h = selection_hash(cell_ids)
    commit = [r for r in log if r["event"] == "selection_committed" and r["output"]["plan_id"] == plan_id]
    if not commit:
        raise ApprovalError(f"plan {plan_id}: selection not committed to the log before reveal")
    if commit[-1]["output"]["selection_hash"] != h:
        raise ApprovalError(f"plan {plan_id}: requested cells differ from the committed selection")
    appr = [r for r in log if r["event"] == "approval" and r["output"].get("plan_id") == plan_id]
    if not appr or appr[-1]["output"].get("decision") != "approved":
        raise ApprovalError(f"plan {plan_id}: no human approval on record")


def reveal(run_id: str, batch: str, plan_id: str, cell_ids: list[str]) -> pd.DataFrame:
    _authorized(run_id, plan_id, cell_ids)
    p = path(load_config()["storage"]["hidden_db"])
    with connect_ro(p) as con:
        df = con.execute(
            f"SELECT cell_id, batch, cycle_life, censored, label_known, long_lived FROM labels_hidden "
            f"WHERE batch = ? AND cell_id IN ({', '.join('?' * len(cell_ids))}) ORDER BY cell_id",
            [batch, *cell_ids],
        ).df()
    research_log.append(run_id, "runner", "reveal", {"plan_id": plan_id, "outcomes": df.to_dict("records")},
                        batch=batch, inputs=cell_ids)
    return df
