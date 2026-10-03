"""Cross-process safe DuckDB access.

DuckDB lets only one process hold a database file open for writing. Under
Omnigent, tools run in subprocesses and policies run in the server process, so
several processes touch the research log at once. `locked()` serialises access
with an exclusive flock on a sidecar `<db>.lock` file, which also makes the
log's read-last-row-then-append step atomic (the hash chain stays consistent).
`connect_ro()` retries read-only opens that collide with a writer.
"""

from __future__ import annotations

import fcntl
import time
from contextlib import contextmanager
from pathlib import Path

import duckdb

LOCK_TIMEOUT_S = 60.0


class DBLockTimeout(TimeoutError):
    pass


@contextmanager
def locked(db_path: Path, timeout: float = LOCK_TIMEOUT_S):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = db_path.with_name(db_path.name + ".lock")
    with open(lock_path, "a") as fh:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise DBLockTimeout(f"could not lock {db_path} within {timeout}s")
                time.sleep(0.02)
        try:
            con = duckdb.connect(str(db_path))
            try:
                yield con
            finally:
                con.close()
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def connect_ro(db_path: Path, timeout: float = LOCK_TIMEOUT_S) -> duckdb.DuckDBPyConnection:
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} not found; run `python -m cyclewise.data.splits` first")
    deadline = time.monotonic() + timeout
    while True:
        try:
            return duckdb.connect(str(db_path), read_only=True)
        except duckdb.IOException:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)
