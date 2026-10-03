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
    """Split the extension budget between the two steps. Each option fixes how many cells
    continue to 100 (m) and how many it plans to carry on to 150 (m150 >= k_final), so the
    150 checkpoint has a real choice (m150 > k_final) unless the option says otherwise."""
    slots = (b.extension - spent) // STEP
    k = b.k_final
    out = []
    # Largest affordable carry to 150 while at least as many cells reach 100: slots // 2.
    deep = max(k, min(n_alive, slots // 2))
    splits = (("A", "wide_shallow", k), ("B", "balanced", k + max(1, (deep - k) // 2)), ("C", "narrow_deep", deep))
    for oid, name, m150 in splits:
        m100 = min(n_alive, slots - m150)
        if m100 < m150 or m150 < k:
            continue
        out.append({"option_id": oid, "name": name, "action": "continue_to_100", "m": m100, "m150_planned": m150,
                    "ranking": "rule", "cost": (m100 + m150) * STEP,
                    "description": f"continue the top {m100} to 100, then plan to carry {m150} to 150 "
                                   f"(margin {m150 - k} over the {k} kept); up to {(m100 + m150) * STEP} channel-cycles"})
    if not out:  # budget cannot carry k_final through both steps: best effort, never over budget
        half = max(0, min(n_alive, slots // 2))
        out.append({"option_id": "A", "name": "best_effort", "action": "continue_to_100", "m": half,
                    "m150_planned": half, "ranking": "rule", "cost": 2 * half * STEP,
                    "description": f"budget too small to carry {k} to 150: continue the top {half} to 100 "
                                   f"and carry them all to 150"})
    return _at_least_two(out)


def options_at_100(b: Budget, n_alive: int, spent: int, planned: int | None = None) -> list[dict]:
    """How many to carry to 150. Never fewer than k_final (pre-registered)."""
    hi = max(0, min(n_alive, (b.extension - spent) // STEP))
    k = min(b.k_final, n_alive, hi)   # the budget is a hard cap; below k_final is a logged best effort
    cands = []
    for oid, name, m in (("A", "planned", planned if planned is not None else hi), ("B", "minimum", k), ("C", "max", hi)):
        m = max(k, min(hi, m))
        if m not in [c["m"] for c in cands]:
            cands.append({"option_id": oid, "name": name, "action": "continue_to_150", "m": m, "ranking": "rule",
                          "cost": m * STEP, "description": f"continue the top {m} to 150 (margin {m - b.k_final}); "
                                                           f"up to {m * STEP} channel-cycles"})
    return _at_least_two(cands)


def options_at_150(b: Budget, n_alive: int) -> list[dict]:
    k = min(b.k_final, n_alive)
    return [
        {"option_id": "A", "name": "rule", "action": "keep_to_eol", "m": k, "ranking": "rule",
         "description": f"keep the top {k} of {n_alive} by the rule in force to end of life"},
        {"option_id": "B", "name": "dq_only", "action": "keep_to_eol", "m": k, "ranking": "dq_only",
         "description": f"keep the top {k} of {n_alive} by ΔQ variance alone (Severson base signal) to end of life"},
    ]


def _at_least_two(opts: list[dict]) -> list[dict]:
    """The brief requires >= 2 options. If sizes collapse, offer the same size ranked by ΔQ alone
    (never a size that breaks the pre-registered k_final rule)."""
    if len(opts) >= 2:
        return opts
    o = dict(opts[0])
    o.update(option_id="D", name=o["name"] + "_dq_only", ranking="dq_only",
             description=o["description"] + "; ranked by ΔQ variance alone")
    return [opts[0], o]


def cut_to_budget(ranked: list[str], m: int, remaining: int) -> tuple[list[str], bool]:
    """Take the top-m of a ranking, cut at the remaining budget. Returns (cells, was_cut)."""
    afford = max(0, remaining // STEP)
    take = min(m, afford, len(ranked))
    return ranked[:take], take < min(m, len(ranked))
