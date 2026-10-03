"""Citations must come from the closed list in CITATIONS.md."""

from __future__ import annotations

import re
from functools import lru_cache

from cyclewise.config import ROOT


@lru_cache(maxsize=None)
def allowed() -> frozenset[str]:
    text = (ROOT / "CITATIONS.md").read_text()
    return frozenset(re.findall(r"^\|\s*([a-z0-9_]+)\s*\|", text, flags=re.M)) - {"id"}


def validate(cites: list[str]) -> list[str]:
    bad = [c for c in cites if c not in allowed()]
    if bad:
        raise ValueError(f"citations {bad} are not in CITATIONS.md; allowed: {sorted(allowed())}")
    return cites
