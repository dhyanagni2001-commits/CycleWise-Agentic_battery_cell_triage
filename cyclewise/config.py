"""Load and validate the run configuration (config/cyclewise.yaml)."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "cyclewise.yaml"
EXCLUSIONS_PATH = ROOT / "config" / "exclusions.yaml"


class ConfigError(ValueError):
    pass


def _validate(cfg: dict[str, Any]) -> None:
    pre = cfg["preregistered"]
    budget = pre["budget_cells"]
    if not isinstance(budget, int) or isinstance(budget, bool):
        raise ConfigError(f"budget_cells must be an integer, got {budget!r}")
    if budget <= 0:
        raise ConfigError(f"budget_cells must be > 0, got {budget}")
    if pre["budget_unit"] != "cells":
        raise ConfigError("budget_unit must be 'cells' (one unit, used consistently)")
    if pre["early_cutoff"] != 50:
        raise ConfigError("early_cutoff is pre-registered at 50")
    lo, hi = pre["delta_q_cycles"]
    if not (pre["window_start"] <= lo < hi <= pre["early_cutoff"]):
        raise ConfigError("delta_q_cycles must lie inside [window_start, early_cutoff]")
    if not 0 < pre["long_lived_quantile"] < 1:
        raise ConfigError("long_lived_quantile must be in (0, 1)")
    if pre["bootstrap_resamples"] < 1000:
        raise ConfigError("bootstrap_resamples must be >= 1000")


@lru_cache(maxsize=None)
def load_config(path: str | None = None) -> dict[str, Any]:
    p = Path(path) if path else CONFIG_PATH
    cfg = yaml.safe_load(p.read_text())
    _validate(cfg)
    cfg["_path"] = str(p)
    cfg["_hash"] = config_hash(cfg)
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    """Hash of what pre-registration protects: the `preregistered` block, the batch
    roles, and the continuation map. Storage paths, URLs, timeouts etc. may change
    without invalidating a frozen run."""
    body = {
        "preregistered": cfg["preregistered"],
        "batches": {k: {"name": b["name"], "role": b["role"]} for k, b in cfg["batches"].items()},
        "continuations": cfg.get("continuations"),
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]


def load_exclusions() -> list[dict[str, str]]:
    return yaml.safe_load(EXCLUSIONS_PATH.read_text())["excluded"]


def path(rel: str) -> Path:
    return ROOT / rel
