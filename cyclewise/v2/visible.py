"""Per-run agent-visible database: only the cycles the lab has paid for.

Each run gets warehouse/v2/runs/<run_id>.duckdb, created as a copy of the
screening DB (cycles <= 50, no labels). `materialize()` is the ONLY way more
data enters it: after a checkpoint decision is approved and committed, it
copies cycles (c, c'] (and the Qdlin curve at c') for exactly the extended
cells. A `paid` table records how far each cell has been paid for, and every
query is checked against it (tool code), on top of the data simply not being
there (physical).
"""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

from cyclewise.config import path
from cyclewise.record import research_log
from cyclewise.record.dblock import connect_ro, locked
from cyclewise.tools.cutoff_view import LeakageError
from cyclewise.v2.data import CKPT_TOL, HIDDEN, SCREEN

RUNS = "warehouse/v2/runs"


def db_path(run_id: str):
    if not run_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError(f"bad run id {run_id!r}")
    return path(RUNS) / f"{run_id}.duckdb"


def create(run_id: str) -> None:
    p = db_path(run_id)
    if p.exists():
        raise FileExistsError(f"visible DB for {run_id} already exists")
    p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path(SCREEN), p)
    with locked(p) as con:
        con.execute("CREATE TABLE paid AS SELECT cell_id, batch, 50 AS paid_to, NULL::INTEGER AS eol "
                    "FROM cells WHERE status = 'eligible'")


def paid(run_id: str, batch: str) -> pd.DataFrame:
    with locked(db_path(run_id)) as con:
        return con.execute("SELECT * FROM paid WHERE batch = ? ORDER BY cell_id", [batch]).df()


def check_access(run_id: str, cell_ids: list[str], max_cycle: int) -> None:
    """Tool-code guard: refuse any request for a cycle a cell has not been paid to."""
    p = paid(run_id, cell_ids[0][:2]).set_index("cell_id") if cell_ids else None
    for c in cell_ids:
        if p is None or c not in p.index:
            raise LeakageError(f"unknown or ineligible cell {c}")
        if max_cycle > int(p.loc[c, "paid_to"]):
            raise LeakageError(f"{c}: cycle {max_cycle} not paid for (paid to {int(p.loc[c, 'paid_to'])})")


def query_cycles(run_id: str, cell_ids: list[str], signals: list[str], max_cycle: int) -> pd.DataFrame:
    check_access(run_id, cell_ids, max_cycle)
    sig = [s for s in signals if s in ("qd", "qc", "ir", "tavg", "tmin", "tmax", "chargetime", "glitch")]
    with locked(db_path(run_id)) as con:
        return con.execute(
            f"SELECT cell_id, cycle, {', '.join(sig)} FROM cycles WHERE cycle <= ? "
            f"AND cell_id IN ({', '.join('?' * len(cell_ids))}) ORDER BY cell_id, cycle",
            [max_cycle, *cell_ids]).df()


def materialize(run_id: str, batch: str, cell_ids: list[str], from_cycle: int, to_cycle: int,
                plan_id: str) -> pd.DataFrame:
    """Pay for cycles (from_cycle, to_cycle] for these cells. Requires a committed, approved
    checkpoint decision for plan_id. Returns per-cell paid_to and EOL if reached in the window."""
    log = research_log.rows(run_id=run_id)
    commit = [r for r in log if r["event"] == "checkpoint_committed" and r["output"]["plan_id"] == plan_id]
    appr = [r for r in log if r["event"] == "approval" and r["output"].get("plan_id") == plan_id]
    if not commit or not appr or appr[-1]["output"].get("decision") != "approved":
        raise PermissionError(f"plan {plan_id}: extension needs a committed and approved decision")
    if sorted(commit[-1]["output"]["extend"]) != sorted(cell_ids):
        raise PermissionError(f"plan {plan_id}: cells differ from the committed extension set")
    if not cell_ids:
        return pd.DataFrame(columns=["cell_id", "paid_to", "eol"])
    ph = ", ".join("?" * len(cell_ids))
    with connect_ro(path(HIDDEN)) as h:
        new_cyc = h.execute(f"SELECT * FROM cycles_raw WHERE cycle > ? AND cycle <= ? AND cell_id IN ({ph})",
                            [from_cycle, to_cycle, *cell_ids]).df()
        new_q = h.execute(f"SELECT * FROM qdlin_ckpt WHERE abs(cycle - ?) <= ? AND cell_id IN ({ph})",
                          [to_cycle, CKPT_TOL, *cell_ids]).df()
        eol = h.execute(f"SELECT cell_id, cycle_life FROM labels WHERE cell_id IN ({ph})", cell_ids).df()
    eol = eol.set_index("cell_id")["cycle_life"]
    out = []
    with locked(db_path(run_id)) as con:
        con.register("nc", new_cyc[new_cyc["cycle"] <= new_cyc["cell_id"].map(eol).fillna(to_cycle)])
        con.execute("INSERT INTO cycles SELECT * FROM nc")
        # Never copy a curve past what was paid for (tolerance is +-2) or past the cell's EOL.
        new_q = new_q[new_q["cycle"] <= np.minimum(to_cycle, new_q["cell_id"].map(eol).fillna(to_cycle))]
        if len(new_q):
            con.register("nq", new_q)
            con.execute("INSERT INTO qdlin SELECT * FROM nq")
        for c in cell_ids:
            # The lab observes EOL only if the cell crosses it inside the paid window.
            died = int(eol[c]) <= to_cycle
            paid_to = int(eol[c]) if died else to_cycle
            con.execute("UPDATE paid SET paid_to = ?, eol = ? WHERE cell_id = ?",
                        [paid_to, int(eol[c]) if died else None, c])
            out.append({"cell_id": c, "paid_to": paid_to, "eol": int(eol[c]) if died else None})
    return pd.DataFrame(out)


def reveal_eol(run_id: str, batch: str, cell_ids: list[str], plan_id: str) -> pd.DataFrame:
    """Final step: the kept cells are cycled to EOL; their full outcome becomes known."""
    log = research_log.rows(run_id=run_id)
    commit = [r for r in log if r["event"] == "checkpoint_committed" and r["output"]["plan_id"] == plan_id]
    appr = [r for r in log if r["event"] == "approval" and r["output"].get("plan_id") == plan_id]
    if not commit or not appr or appr[-1]["output"].get("decision") != "approved":
        raise PermissionError(f"plan {plan_id}: keep-to-EOL needs a committed and approved decision")
    if sorted(commit[-1]["output"]["keep"]) != sorted(cell_ids):
        raise PermissionError(f"plan {plan_id}: cells differ from the committed keep set")
    ph = ", ".join("?" * len(cell_ids))
    with connect_ro(path(HIDDEN)) as h:
        eol = h.execute(f"SELECT cell_id, cycle_life, censored FROM labels WHERE cell_id IN ({ph})", cell_ids).df()
    with locked(db_path(run_id)) as con:
        for r in eol.itertuples():
            con.execute("UPDATE paid SET eol = ?, paid_to = ? WHERE cell_id = ?",
                        [int(r.cycle_life), int(r.cycle_life), r.cell_id])
    return eol


def qdlin_at(run_id: str, cell_ids: list[str], cycle: int) -> dict[str, np.ndarray]:
    """Qdlin at (or within +-2 of) `cycle` for cells paid at least that far."""
    check_access(run_id, cell_ids, cycle)
    ph = ", ".join("?" * len(cell_ids))
    with locked(db_path(run_id)) as con:
        df = con.execute(f"SELECT cell_id, cycle, qdlin FROM qdlin WHERE abs(cycle - ?) <= ? AND cell_id IN ({ph})",
                         [cycle, CKPT_TOL if cycle > 50 else 0, *cell_ids]).df()
    out = {}
    for cid, g in df.groupby("cell_id"):
        g = g.assign(d=(g["cycle"] - cycle).abs()).sort_values(["d", "cycle"])
        out[cid] = np.asarray(g["qdlin"].iloc[0], dtype=float)
    return out
