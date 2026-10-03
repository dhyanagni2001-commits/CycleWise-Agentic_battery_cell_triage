"""Build the two-database warehouse that mirrors the Unity Catalog split.

  warehouse/early.duckdb   (agent-visible)
      cells            id, batch, protocol, QC status. No labels.
      cycles_early     summary signals, cycle_index <= 50 ONLY (physically truncated)
      qdlin_early      Qdlin(V) curves, cycles 2..50
      features_early   per-cell early features
      voltage_grid

  warehouse/hidden.duckdb  (never opened by agent tools; only reveal() and the scorer)
      cycles_raw       all cycles
      labels_hidden    cycle life, censoring, long-lived label
      frozen_params    the long-lived threshold, frozen once

On Databricks the same split is a UC view `cycles_early` + grants (see
databricks/01_unity_catalog.sql). Locally the split is physical: the agent DB
file does not contain a single row past cycle 50 or a single label.

Run:  python -m cyclewise.data.splits
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd
import yaml

from cyclewise.config import ROOT, load_config, path
from cyclewise.data.featurize import build_features
from cyclewise.data.load_raw import STAGING

FROZEN_PATH = ROOT / "config" / "frozen.yaml"


def long_lived_threshold(labels: pd.DataFrame, cfg: dict) -> float:
    """Pre-registered rule: quantile of TRAINING-batch cycle life (censored = lower bound)."""
    pre = cfg["preregistered"]
    train = [k for k, b in cfg["batches"].items() if b["role"] == "train"]
    cl = labels[labels["batch"].isin(train)]["cycle_life"].astype(float)
    return float(np.quantile(cl, pre["long_lived_quantile"]))


def label_long_lived(labels: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """>= threshold is long-lived. A censored cell whose lower bound is below the
    threshold has an unknown label (excluded from scoring and counted)."""
    out = labels.copy()
    known = ~out["censored"] | (out["cycle_life"] >= threshold)
    out["label_known"] = known
    out["long_lived"] = np.where(known, out["cycle_life"] >= threshold, None)
    return out


def freeze(cfg: dict, labels: pd.DataFrame) -> dict:
    """Freeze the threshold once. If already frozen, re-use it and never recompute."""
    if FROZEN_PATH.exists():
        frozen = yaml.safe_load(FROZEN_PATH.read_text())
        if frozen["config_hash"] != cfg["_hash"]:
            raise RuntimeError(
                "the pre-registered part of config/cyclewise.yaml changed after the threshold was frozen "
                f"(frozen hash {frozen['config_hash']}, now {cfg['_hash']}). Revert the change. "
                "Only if this is a NEW pre-registered study, delete config/frozen.yaml deliberately and "
                "re-run `python -m cyclewise.data.splits`; earlier runs are then not comparable.")
        return frozen
    thr = long_lived_threshold(labels, cfg)
    frozen = {
        "long_lived_threshold_cycles": thr,
        "rule": cfg["preregistered"]["long_lived_rule"],
        "quantile": cfg["preregistered"]["long_lived_quantile"],
        "computed_on": [k for k, b in cfg["batches"].items() if b["role"] == "train"],
        "config_hash": cfg["_hash"],
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    FROZEN_PATH.write_text(
        "# Written once by cyclewise.data.splits before any agent run. Do not edit.\n"
        + yaml.safe_dump(frozen, sort_keys=False)
    )
    return frozen


def build(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    pre = cfg["preregistered"]
    cutoff = pre["early_cutoff"]
    st = path(STAGING)
    cells = pd.read_parquet(st / "cells.parquet")
    cycles_raw = pd.read_parquet(st / "cycles_raw.parquet")
    qdlin = pd.read_parquet(st / "qdlin_early.parquet")
    labels = pd.read_parquet(st / "labels.parquet")
    vgrid = pd.read_parquet(st / "vgrid.parquet")

    cycles_early = cycles_raw[cycles_raw["cycle"] <= cutoff].copy()
    feats = build_features(cells, cycles_early, qdlin, cfg)

    frozen = freeze(cfg, labels)
    labels = label_long_lived(labels, frozen["long_lived_threshold_cycles"])

    early_db, hidden_db = path(cfg["storage"]["early_db"]), path(cfg["storage"]["hidden_db"])
    early_db.parent.mkdir(parents=True, exist_ok=True)
    for p in (early_db, hidden_db):
        if p.exists():
            p.unlink()

    qd = qdlin.copy()
    qd["qdlin"] = qd["qdlin"].apply(lambda a: [float(x) for x in a])
    agent_cells = cells.drop(columns=[c for c in cells.columns if "life" in c], errors="ignore")
    with duckdb.connect(str(early_db)) as con:
        con.register("c", agent_cells)
        con.execute("CREATE TABLE cells AS SELECT * FROM c")
        con.register("y", cycles_early)
        con.execute("CREATE TABLE cycles_early_t AS SELECT * FROM y")
        # The view is what tools query; the CHECK documents the contract.
        con.execute(f"CREATE VIEW cycles_early AS SELECT * FROM cycles_early_t WHERE cycle <= {cutoff}")
        con.register("q", qd)
        con.execute("CREATE TABLE qdlin_early AS SELECT * FROM q")
        con.register("f", feats)
        con.execute("CREATE TABLE features_early AS SELECT * FROM f")
        con.register("v", vgrid)
        con.execute("CREATE TABLE voltage_grid AS SELECT * FROM v")
        max_c = con.execute("SELECT max(cycle) FROM cycles_early_t").fetchone()[0]
        assert max_c <= cutoff
        tables = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
        assert not tables & {"labels_hidden", "cycles_raw", "frozen_params"}

    with duckdb.connect(str(hidden_db)) as con:
        con.register("r", cycles_raw)
        con.execute("CREATE TABLE cycles_raw AS SELECT * FROM r")
        lab = labels.copy()
        lab["long_lived"] = lab["long_lived"].astype("object")
        con.register("l", lab)
        con.execute("CREATE TABLE labels_hidden AS SELECT * FROM l")
        con.execute("CREATE TABLE frozen_params (k VARCHAR, v VARCHAR)")
        con.execute("INSERT INTO frozen_params VALUES (?, ?)", ["frozen", json.dumps(frozen)])

    # Parquet copies for the Databricks upload script.
    for name, df in {"cells": agent_cells, "cycles_early": cycles_early, "features_early": feats,
                     "cycles_raw": cycles_raw, "labels_hidden": labels}.items():
        df.to_parquet(st / f"delta_{name}.parquet", index=False)

    counts = {
        b: {
            "cells": int((cells["batch"] == b).sum()),
            "eligible": int(((cells["batch"] == b) & (cells["status"] == "eligible")).sum()),
            "censored": int(labels[labels["batch"] == b]["censored"].sum()),
            "label_unknown": int((~labels[labels["batch"] == b]["label_known"]).sum()),
            "features_nan_cells": int(feats[feats["batch"] == b].isna().any(axis=1).sum()),
        }
        for b in cfg["batches"]
    }
    return {"frozen": frozen, "counts": counts}


def main() -> None:
    out = build()
    print("frozen long-lived threshold written to config/frozen.yaml (value not printed)")
    print(json.dumps(out["counts"], indent=2))


if __name__ == "__main__":
    main()
