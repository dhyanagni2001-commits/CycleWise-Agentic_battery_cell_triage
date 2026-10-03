"""Budget checks. Budget unit is CELLS (integer), used everywhere."""

from __future__ import annotations

from cyclewise.config import load_config


class BudgetError(ValueError):
    pass


def total_budget() -> int:
    return load_config()["preregistered"]["budget_cells"]


def check(n_cells: int, remaining: int, eligible: int) -> dict:
    """Accept a request for n_cells, or raise BudgetError. Returns a status dict."""
    if not isinstance(n_cells, int) or isinstance(n_cells, bool) or n_cells <= 0:
        raise BudgetError(f"request must be a positive integer number of cells, got {n_cells!r}")
    if n_cells > remaining:
        raise BudgetError(f"requested {n_cells} cells but only {remaining} remain in the budget")
    warning = ""
    if remaining >= eligible:
        warning = (f"budget ({remaining}) >= eligible cells ({eligible}): selecting all; "
                   "recall is trivially 100%")
    return {"ok": True, "requested": n_cells, "remaining_after": remaining - n_cells, "warning": warning}
