"""Budgets and checkpoint options (all numbers from config/prereg_v2.yaml).

Budget unit: channel-cycles. k_final = round(0.25 N) cells are kept to EOL;
extensions 50->100 and 100->150 share extension_budget = 50 * ceil(0.75 N),
and at least k_final cells must reach 150. Options are generated here, inside
the budget, so a Planner can only choose among affordable allocations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

STEP = 50  # cycles per extension (50->100, 100->150)


@dataclass(frozen=True)
class Budget:
    n: int
    k_final: int
    extension: int

    @classmethod
    def for_batch(cls, n_eligible: int) -> "Budget":
        if n_eligible <= 0:
            raise ValueError("no eligible cells")
        return cls(n=n_eligible, k_final=max(1, round(0.25 * n_eligible)),
                   extension=STEP * math.ceil(0.75 * n_eligible))


def options_at_50(b: Budget, n_alive: int, spent: int) -> list[dict]:
    """How many cells to continue to 100. Reserve k_final extensions for 100->150."""
    remaining = b.extension - spent
    hi = min(n_alive, remaining // STEP - b.k_final)
    if hi < b.k_final:
        hi = min(n_alive, remaining // STEP)  # cannot reserve; continue what we can
    cands = {
        "A": ("narrow", min(hi, b.k_final + max(1, b.k_final // 2))),
        "B": ("balanced", min(hi, 2 * b.k_final)),
        "C": ("wide", hi),
    }
    return _dedupe(cands, "continue_to_100",
                   "keep the top-m cells by score on test to cycle 100; stop the rest at 50")


def options_at_100(b: Budget, n_alive: int, spent: int) -> list[dict]:
    remaining = b.extension - spent
    hi = min(n_alive, remaining // STEP)
    lo = min(hi, b.k_final)
    cands = {
        "A": ("exploit", lo),
        "B": ("margin", min(hi, lo + max(1, b.k_final // 2))),
        "C": ("max", hi),
    }
    return _dedupe(cands, "continue_to_150", "keep the top-m cells by score on test to cycle 150")


def options_at_150(b: Budget, n_alive: int) -> list[dict]:
    k = min(b.k_final, n_alive)
    return [
        {"option_id": "A", "name": "rule", "action": "keep_to_eol", "m": k, "ranking": "rule",
         "description": f"keep the top {k} by the rule in force to end of life"},
        {"option_id": "B", "name": "dq_only", "action": "keep_to_eol", "m": k, "ranking": "dq_only",
         "description": f"keep the top {k} by ΔQ variance alone (Severson base signal) to end of life"},
    ]


def _dedupe(cands: dict, action: str, desc: str) -> list[dict]:
    out, seen = [], set()
    for oid, (name, m) in cands.items():
        m = max(0, int(m))
        if m in seen:
            continue
        seen.add(m)
        out.append({"option_id": oid, "name": name, "action": action, "m": m, "ranking": "rule",
                    "cost": m * STEP, "description": f"{desc} (m={m}, up to {m * STEP} channel-cycles)"})
    if len(out) < 2:  # always offer at least two distinct options
        m = out[0]["m"]
        alt = max(0, m - 1)
        out.append({"option_id": "Z", "name": "one_fewer", "action": action, "m": alt, "ranking": "rule",
                    "cost": alt * STEP, "description": f"{desc} (m={alt})"})
    return out


def cut_to_budget(ranked: list[str], m: int, remaining: int) -> tuple[list[str], bool]:
    """Take the top-m of a ranking, cut at the remaining budget. Returns (cells, was_cut)."""
    afford = max(0, remaining // STEP)
    take = min(m, afford, len(ranked))
    return ranked[:take], take < min(m, len(ranked))
