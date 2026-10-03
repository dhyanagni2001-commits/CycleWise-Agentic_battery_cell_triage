"""Append-only, hash-chained research log.

Every agent step, approval, selection commit, reveal, and metric is one row.
Rows are never updated or deleted: the module exposes `append` and readers only.
Each row stores the SHA-256 of the previous row, so any edit breaks `verify()`.

Local backend: DuckDB (warehouse/research_log.duckdb).
Databricks backend: the same rows are mirrored to the Delta table
`research_log` by databricks/upload_tables.py.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

import numpy as np

from cyclewise.config import load_config, path
from cyclewise.record.dblock import locked

SCHEMA = """
CREATE TABLE IF NOT EXISTS research_log (
    seq BIGINT PRIMARY KEY,
    ts VARCHAR,
    run_id VARCHAR,
    batch VARCHAR,
    agent VARCHAR,
    event VARCHAR,
    input_hash VARCHAR,
    output_json VARCHAR,
    model VARCHAR,
    prompt_version VARCHAR,
    fallback_used BOOLEAN,
    prev_hash VARCHAR,
    row_hash VARCHAR
)
"""


@contextmanager
def _db():
    """Exclusive, cross-process access to the log (see dblock)."""
    with locked(path(load_config()["storage"]["log_db"])) as con:
        con.execute(SCHEMA)
        yield con


def _jsonable(o: Any) -> Any:
    """numpy / pandas scalars -> plain Python, so values round-trip with their type."""
    if isinstance(o, np.generic):
        return o.item()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=_jsonable)


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(_dumps(obj).encode()).hexdigest()


def append(run_id: str, agent: str, event: str, output: Any, *, batch: str = "",
           inputs: Any = None, model: str = "", prompt_version: str = "",
           fallback_used: bool = False) -> dict:
    with _db() as con:
        last = con.execute("SELECT seq, row_hash FROM research_log ORDER BY seq DESC LIMIT 1").fetchone()
        seq, prev = (last[0] + 1, last[1]) if last else (1, "genesis")
        row = {
            "seq": seq,
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "batch": batch,
            "agent": agent,
            "event": event,
            "input_hash": stable_hash(inputs)[:16] if inputs is not None else "",
            "output_json": _dumps(output),
            "model": model,
            "prompt_version": prompt_version,
            "fallback_used": bool(fallback_used),
            "prev_hash": prev,
        }
        row["row_hash"] = stable_hash(row)
        con.execute(f"INSERT INTO research_log VALUES ({', '.join('?' * len(row))})", list(row.values()))
    return row


def rows(run_id: str | None = None, event: str | None = None, batch: str | None = None) -> list[dict]:
    q, args = "SELECT * FROM research_log WHERE 1=1", []
    for col, val in (("run_id", run_id), ("event", event), ("batch", batch)):
        if val is not None:
            q += f" AND {col} = ?"
            args.append(val)
    with _db() as con:
        df = con.execute(q + " ORDER BY seq", args).df()
    out = df.to_dict("records")
    for r in out:
        r["output"] = json.loads(r["output_json"])
    return out


def verify() -> bool:
    """Recompute the hash chain. False if any row was edited, deleted, or reordered."""
    with _db() as con:
        df = con.execute("SELECT * FROM research_log ORDER BY seq").df()
    prev = "genesis"
    for r in df.to_dict("records"):
        stored = r.pop("row_hash")
        r["fallback_used"] = bool(r["fallback_used"])
        r["seq"] = int(r["seq"])
        if r["prev_hash"] != prev or stable_hash(r) != stored:
            return False
        prev = stored
    return True
