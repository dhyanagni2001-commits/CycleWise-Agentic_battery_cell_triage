"""The only data access agents have: read-only queries against the early DB.

Defense in depth for the cycle-50 cutoff:
  1. Physical: early.duckdb holds no row past cycle 50 and no labels
     (on Databricks: UC view + no SELECT grant on labels/raw tables).
  2. Tool code: every argument naming a cycle is checked here; any value
     above the cutoff raises LeakageError before a query runs.
  3. Omnigent policy: cyclewise.policies.omni_policies.leakage_guard denies
     tool calls whose arguments reference a cycle above 50 or a hidden table.
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

from cyclewise.config import load_config, path
from cyclewise.record.dblock import connect_ro

FORBIDDEN_TABLES = ("labels_hidden", "cycles_raw", "frozen_params", "hidden")
ALLOWED_SIGNALS = ("qd", "qc", "ir", "tavg", "tmin", "tmax", "chargetime", "glitch")


class LeakageError(PermissionError):
    pass


def cutoff() -> int:
    return load_config()["preregistered"]["early_cutoff"]


def check_cycle(max_cycle: int | None) -> int:
    c = cutoff()
    if max_cycle is None:
        return c
    if not isinstance(max_cycle, int) or isinstance(max_cycle, bool):
        raise LeakageError(f"max_cycle must be an integer, got {max_cycle!r}")
    if max_cycle > c:
        raise LeakageError(f"cycle {max_cycle} is past the early cutoff ({c}); access denied")
    return max_cycle


def _con():
    p = path(load_config()["storage"]["early_db"])
    if any(t in p.name for t in FORBIDDEN_TABLES):
        raise LeakageError("tool attempted to open a hidden database")
    return connect_ro(p)


def _ids(cell_ids: Iterable[str] | None) -> list[str] | None:
    if cell_ids is None:
        return None
    ids = [str(c) for c in cell_ids]
    for c in ids:
        if not re.fullmatch(r"b\dc\d+", c):
            raise ValueError(f"malformed cell id {c!r}")
    return ids


def list_cells(batch: str, status: str | None = "eligible") -> pd.DataFrame:
    q, args = "SELECT cell_id, batch, protocol, status, status_reason FROM cells WHERE batch = ?", [batch]
    if status:
        q += " AND status = ?"
        args.append(status)
    with _con() as con:
        return con.execute(q + " ORDER BY cell_id", args).df()


def query_early(cell_ids: list[str] | None = None, signals: list[str] | None = None,
                min_cycle: int = 1, max_cycle: int | None = None, batch: str | None = None) -> pd.DataFrame:
    """Per-cycle summary signals for cycles min_cycle..max_cycle (max 50)."""
    hi = check_cycle(max_cycle)
    check_cycle(min_cycle)
    sig = signals or ["qd", "ir", "tmax", "chargetime"]
    bad = [s for s in sig if s not in ALLOWED_SIGNALS]
    if bad:
        raise ValueError(f"unknown signals {bad}; allowed {ALLOWED_SIGNALS}")
    q = f"SELECT cell_id, batch, cycle, {', '.join(sig)} FROM cycles_early WHERE cycle BETWEEN ? AND ?"
    args: list = [min_cycle, hi]
    ids = _ids(cell_ids)
    if ids:
        q += f" AND cell_id IN ({', '.join('?' * len(ids))})"
        args += ids
    if batch:
        q += " AND batch = ?"
        args.append(batch)
    with _con() as con:
        df = con.execute(q + " ORDER BY cell_id, cycle", args).df()
    assert df.empty or df["cycle"].max() <= cutoff()
    return df


def get_features(batch: str, features: list[str] | None = None,
                 cell_ids: list[str] | None = None) -> pd.DataFrame:
    with _con() as con:
        df = con.execute("SELECT * FROM features_early WHERE batch = ? ORDER BY cell_id", [batch]).df()
    if cell_ids:
        df = df[df["cell_id"].isin(_ids(cell_ids))]
    if features:
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f"unknown features {missing}")
        df = df[["cell_id", "batch", *features]]
    return df.reset_index(drop=True)


def feature_summary(batch: str) -> dict:
    """Label-free distribution summary the Evidence agent can reason over."""
    df = get_features(batch).drop(columns=["batch"]).set_index("cell_id")
    return df.describe().T[["mean", "std", "min", "50%", "max"]].round(5).to_dict("index")
