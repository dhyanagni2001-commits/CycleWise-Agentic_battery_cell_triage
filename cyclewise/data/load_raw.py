"""Load the Severson et al. 2019 MATLAB v7.3 (HDF5) batch files into tidy tables.

Outputs (Parquet, under warehouse/staging/):
  cells.parquet         one row per cell: id, batch, protocol, QC counts and flags
  cycles_raw.parquet    one row per (cell, cycle): summary signals, glitch flag
  qdlin_early.parquet   Qdlin(V) curves for cycles <= early_cutoff only
  labels.parquet        cycle life, censoring, file-reported cycle life

Quirks handled (spec 2 / 4.1):
  * HDF5 object references dereferenced; strings stored as uint16 arrays decoded.
  * Batch 1 cells continued in batch 2 are concatenated and the batch 2 copies dropped.
  * Non-finite core fields drop the cycle; forward-fill limited to 2 cycles.
  * Capacity glitches flagged with a rolling-median filter, excluded from features and EOL.
  * Cells that never reach EOL are right-censored, not dropped.
  * Value ranges asserted on load (Ah, degC, ohm).

Run:  python -m cyclewise.data.load_raw
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from cyclewise.config import load_config, load_exclusions, path

log = logging.getLogger(__name__)

SUMMARY_FIELDS = {
    "QDischarge": "qd",
    "QCharge": "qc",
    "IR": "ir",
    "Tavg": "tavg",
    "Tmin": "tmin",
    "Tmax": "tmax",
    "chargetime": "chargetime",
}
CORE_FIELDS = ["qd"]  # a cycle without these is dropped
STAGING = "warehouse/staging"


def _decode(f: h5py.File, ref) -> str:
    """MATLAB char arrays are stored as uint16 code units."""
    arr = f[ref][()]
    return arr.astype(np.uint16).tobytes().decode("utf-16-le").replace("\x00", "").strip()


@dataclass
class RawCell:
    cell_id: str
    batch: str
    protocol: str
    channel_id: str
    barcode: str
    file_cycle_life: float
    summary: pd.DataFrame                  # one row per cycle, columns cycle + SUMMARY_FIELDS
    qdlin: dict[int, np.ndarray] = field(default_factory=dict)  # cycle -> Qdlin (<= cutoff)
    notes: list[str] = field(default_factory=list)


def read_batch(file: Path, batch_key: str, early_cutoff: int) -> tuple[dict[str, RawCell], np.ndarray]:
    """Read one batch file. Detailed Qdlin curves are read only for cycles <= early_cutoff."""
    cells: dict[str, RawCell] = {}
    with h5py.File(file, "r") as f:
        b = f["batch"]
        vdlin = np.asarray(f[b["Vdlin"][0, 0]][()]).ravel()
        n = b["summary"].shape[0]
        for i in range(n):
            cid = f"{batch_key}c{i}"
            s = f[b["summary"][i, 0]]
            data = {"cycle": np.asarray(s["cycle"][0, :], dtype=float)}
            for src, dst in SUMMARY_FIELDS.items():
                data[dst] = np.asarray(s[src][0, :], dtype=float)
            summary = pd.DataFrame(data)
            cyc = f[b["cycles"][i, 0]]
            n_detail = cyc["Qdlin"].shape[0]
            qdlin: dict[int, np.ndarray] = {}
            notes: list[str] = []
            if n_detail != len(summary):
                notes.append(f"detail/summary length mismatch {n_detail} vs {len(summary)}")
            # Detailed cycle j corresponds to summary row j.
            for j in range(min(n_detail, len(summary))):
                c_idx = int(summary["cycle"].iloc[j])
                if c_idx > early_cutoff:
                    break
                qdlin[c_idx] = np.asarray(f[cyc["Qdlin"][j, 0]][()], dtype=float).ravel()
            cl = np.asarray(f[b["cycle_life"][i, 0]][()], dtype=float).ravel()
            cells[cid] = RawCell(
                cell_id=cid,
                batch=batch_key,
                protocol=_decode(f, b["policy_readable"][i, 0]),
                channel_id=_decode(f, b["channel_id"][i, 0]),
                barcode=_decode(f, b["barcode"][i, 0]),
                file_cycle_life=float(cl[0]) if cl.size and np.isfinite(cl[0]) else float("nan"),
                summary=summary,
                qdlin=qdlin,
                notes=notes,
            )
    return cells, vdlin


def merge_continuations(b1: dict[str, RawCell], b2: dict[str, RawCell], pairs: list[dict]) -> None:
    """Append batch 2 continuation data to the batch 1 cell; drop the batch 2 copy."""
    for p in pairs:
        a, c = b1[p["b1"]], b2.pop(p["b2"])
        offset = a.summary["cycle"].max()
        cont = c.summary.copy()
        # Renumber contiguously after the last batch 1 cycle (authors' loader does the same).
        cont["cycle"] = np.arange(1, len(cont) + 1) + offset
        a.summary = pd.concat([a.summary, cont], ignore_index=True)
        a.notes.append(f"continued in {c.cell_id}: +{len(cont)} cycles appended")
    ids = list(b1) + list(b2)
    assert len(ids) == len(set(ids)), "cell id appears twice after merge"


def clean_summary(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Drop broken cycles, flag glitches, limit forward fill. Returns (df, qc_counts)."""
    cl = cfg["cleaning"]
    qc = {"dropped_nonfinite_core": 0, "dropped_dup_cycle": 0, "ffilled_values": 0,
          "unfillable_values": 0, "glitches": 0, "dropped_cycle0": 0}
    df = df.copy()
    n0 = len(df)
    df = df.drop_duplicates(subset="cycle", keep="first")
    qc["dropped_dup_cycle"] = n0 - len(df)
    # Some files carry a cycle 0; feature windows start at window_start anyway.
    qc["dropped_cycle0"] = int((df["cycle"] < 1).sum())
    df = df[df["cycle"] >= 1]
    df = df.sort_values("cycle").reset_index(drop=True)

    core_bad = ~np.isfinite(df[CORE_FIELDS]).all(axis=1)
    qc["dropped_nonfinite_core"] = int(core_bad.sum())
    df = df[~core_bad].reset_index(drop=True)

    # Non-core fields: forward fill at most max_forward_fill consecutive cycles.
    other = [c for c in SUMMARY_FIELDS.values() if c not in CORE_FIELDS]
    for c in other:
        bad = ~np.isfinite(df[c])
        if bad.any():
            filled = df[c].where(~bad).ffill(limit=cl["max_forward_fill"])
            qc["ffilled_values"] += int((bad & filled.notna()).sum())
            qc["unfillable_values"] += int(filled.isna().sum())
            df[c] = filled

    # Capacity glitches: far from the rolling median, above nominal by a wide margin, or <= 0.
    med = df["qd"].rolling(cl["glitch_rolling_window"], center=True, min_periods=1).median()
    lo, hi = cl["capacity_range_ah"]
    glitch = ((df["qd"] - med).abs() > cl["glitch_abs_tol_ah"]) | (df["qd"] > hi) | (df["qd"] <= lo)
    df["glitch"] = glitch
    qc["glitches"] = int(glitch.sum())
    return df, qc


def compute_label(df: pd.DataFrame, eol: float, stop_tol: float) -> dict:
    """Cycle life = first non-glitch cycle with qd < eol.

    Batch 1 recordings stop on the cycle *before* the crossing (the file's own
    cycle_life is last_cycle + 1). So if the last recorded capacity is within
    stop_tol of eol, the test was stopped by the EOL criterion and cycle life is
    last_cycle + 1. Otherwise the cell is right-censored at last_cycle.
    """
    good = df[~df["glitch"]]
    last = int(df["cycle"].max())
    below = good[good["qd"] < eol]
    if len(below):
        return {"cycle_life": int(below["cycle"].iloc[0]), "censored": False,
                "last_cycle": last, "label_source": "observed_crossing"}
    if good["qd"].iloc[-1] <= eol + stop_tol:
        return {"cycle_life": last + 1, "censored": False,
                "last_cycle": last, "label_source": "stopped_at_eol"}
    return {"cycle_life": last, "censored": True,
            "last_cycle": last, "label_source": "censored_lower_bound"}


def assert_ranges(df: pd.DataFrame, cfg: dict, cell_id: str) -> list[str]:
    """Unit checks. Returns warnings; raises if the bulk of a signal is out of range."""
    cl, warn = cfg["cleaning"], []
    good = df[~df["glitch"]]
    checks = {"qd": cl["capacity_range_ah"], "tmax": cl["temperature_range_c"], "ir": cl["ir_range_ohm"]}
    for col, (lo, hi) in checks.items():
        v = good[col].dropna()
        if v.empty:
            warn.append(f"{col}: no finite values")
            continue
        frac_out = float(((v < lo) | (v > hi)).mean())
        if frac_out > 0.5:
            raise AssertionError(f"{cell_id}: {col} mostly outside [{lo}, {hi}] - wrong units?")
        if frac_out > 0:
            warn.append(f"{col}: {frac_out:.1%} of cycles outside [{lo}, {hi}]")
    return warn


def build(cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    from cyclewise.data.download import missing
    if missing():
        raise SystemExit(f"raw data missing or incomplete: {missing()}\n"
                         "Download it with:  python -m cyclewise.data.download")
    pre = cfg["preregistered"]
    cutoff = pre["early_cutoff"]
    excluded = {e["cell_id"]: e["reason"] for e in load_exclusions()}

    raw: dict[str, dict[str, RawCell]] = {}
    grids = {}
    for key, b in cfg["batches"].items():
        log.info("reading %s", b["file"])
        raw[key], grids[key] = read_batch(path(b["file"]), key, cutoff)
    if "b1" in raw and "b2" in raw:
        merge_continuations(raw["b1"], raw["b2"], cfg["continuations"])

    vgrid = grids["b1"]
    for k, g in grids.items():
        assert np.allclose(g, vgrid), f"voltage grid differs in {k}"

    cell_rows, cyc_frames, q_rows, label_rows = [], [], [], []
    for key, cells in raw.items():
        for cid, c in cells.items():
            if cid in excluded:
                cell_rows.append({"cell_id": cid, "batch": key, "protocol": c.protocol,
                                  "status": "excluded", "status_reason": excluded[cid]})
                continue
            df, qc = clean_summary(c.summary, cfg)
            warn = assert_ranges(df, cfg, cid)
            n_early = int(((df["cycle"] >= pre["window_start"]) & (df["cycle"] <= cutoff)).sum())
            expected = cutoff - pre["window_start"] + 1
            assert df["cycle"].max() >= cutoff, f"{cid} has fewer than {cutoff} cycles recorded"
            missing_frac = 1 - n_early / expected
            status, reason = "eligible", ""
            if missing_frac > cfg["cleaning"]["max_missing_window_frac"]:
                status, reason = "insufficient_early_data", f"{missing_frac:.0%} of cycles 2..{cutoff} missing"
            for q_cycle in pre["delta_q_cycles"]:
                if q_cycle not in c.qdlin:
                    status, reason = "insufficient_early_data", f"Qdlin missing at cycle {q_cycle}"
            lab = compute_label(df, pre["eol_capacity_ah"], cfg["cleaning"]["eol_stop_tol_ah"])
            cell_rows.append({
                "cell_id": cid, "batch": key, "protocol": c.protocol, "channel_id": c.channel_id,
                "status": status, "status_reason": reason,
                "n_cycles": int(len(df)), "early_missing_frac": round(missing_frac, 4),
                "notes": "; ".join(c.notes + warn), **{f"qc_{k}": v for k, v in qc.items()},
            })
            df.insert(0, "cell_id", cid)
            df.insert(1, "batch", key)
            cyc_frames.append(df)
            for cyc_idx, q in c.qdlin.items():
                if pre["window_start"] <= cyc_idx <= cutoff and q.size == vgrid.size:
                    q_rows.append({"cell_id": cid, "batch": key, "cycle": int(cyc_idx), "qdlin": q.astype(np.float32)})
            label_rows.append({"cell_id": cid, "batch": key, **lab,
                               "file_cycle_life": c.file_cycle_life})

    out = {
        "cells": pd.DataFrame(cell_rows),
        "cycles_raw": pd.concat(cyc_frames, ignore_index=True),
        "qdlin_early": pd.DataFrame(q_rows),
        "labels": pd.DataFrame(label_rows),
        "vgrid": pd.DataFrame({"i": np.arange(vgrid.size), "voltage": vgrid}),
    }
    out["cycles_raw"]["cycle"] = out["cycles_raw"]["cycle"].astype(int)
    stage = path(STAGING)
    stage.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.to_parquet(stage / f"{name}.parquet", index=False)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = build()
    cells = out["cells"]
    print(cells.groupby(["batch", "status"]).size().to_string())
    print(f"cycles_raw rows: {len(out['cycles_raw'])}, qdlin_early rows: {len(out['qdlin_early'])}")
    print("Labels written to the hidden staging table; counts only:")
    lab = out["labels"]
    print(lab.groupby("batch")["censored"].agg(["size", "sum"]).rename(columns={"sum": "censored"}).to_string())


if __name__ == "__main__":
    main()
