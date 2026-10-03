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
    out = {"prereg_sha256": locked, "prereg_commit": h, "prereg_committed_at": ts}
    if AMEND.exists():
        lock = AMEND.with_suffix(".lock")
        a_locked = lock.read_text().split()[0] if lock.exists() else ""
        if hashlib.sha256(AMEND.read_bytes()).hexdigest() != a_locked:
            raise PreregError("prereg_v2_amendment1.yaml differs from its frozen hash")
        a_commit = _git("log", "-n", "1", "--format=%H %cI", "--", str(AMEND.relative_to(ROOT)))
        if not a_commit or _git("status", "--porcelain", "--", str(AMEND.relative_to(ROOT)), str(lock.relative_to(ROOT))):
            raise PreregError("prereg_v2_amendment1.yaml is not committed or has uncommitted changes")
        ah, ats = a_commit.split(" ", 1)
        out.update(amendment1_sha256=a_locked, amendment1_commit=ah, amendment1_committed_at=ats)
    return out


@lru_cache(maxsize=None)
def load() -> dict:
    return yaml.safe_load(PREREG.read_text())


AMEND = ROOT / "config" / "prereg_v2_amendment1.yaml"


def amendment() -> dict | None:
    """Amendment 1 (written after the b1/b2 dev run, before batch 3 was opened), if present."""
    return yaml.safe_load(AMEND.read_text()) if AMEND.exists() else None
