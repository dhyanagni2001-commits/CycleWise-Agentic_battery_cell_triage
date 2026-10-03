"""v2 pre-registration guard.

config/prereg_v2.yaml was committed to git before batch 3 was downloaded.
`verify()` checks that (1) the file still matches the SHA-256 frozen in
config/prereg_v2.lock and (2) it is committed and unmodified in git. Every v2
entry point that touches batch 3 calls it first and records the commit hash.
"""

from __future__ import annotations

import hashlib
import subprocess
from functools import lru_cache

import yaml

from cyclewise.config import ROOT

PREREG = ROOT / "config" / "prereg_v2.yaml"
LOCK = ROOT / "config" / "prereg_v2.lock"


class PreregError(RuntimeError):
    pass


def sha256() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()


def verify() -> dict:
    locked = LOCK.read_text().split()[0] if LOCK.exists() else ""
    if sha256() != locked:
        raise PreregError("config/prereg_v2.yaml differs from its frozen hash in config/prereg_v2.lock; "
                          "the v2 pre-registration must not change after commit")
    commit = _git("log", "-n", "1", "--format=%H %cI", "--", str(PREREG.relative_to(ROOT)))
    if not commit:
        raise PreregError("config/prereg_v2.yaml is not committed to git; commit it before touching batch 3")
    if _git("status", "--porcelain", "--", str(PREREG.relative_to(ROOT)), str(LOCK.relative_to(ROOT))):
        raise PreregError("config/prereg_v2.yaml or its lock has uncommitted changes")
    h, ts = commit.split(" ", 1)
    return {"prereg_sha256": locked, "prereg_commit": h, "prereg_committed_at": ts}


@lru_cache(maxsize=None)
def load() -> dict:
    return yaml.safe_load(PREREG.read_text())
