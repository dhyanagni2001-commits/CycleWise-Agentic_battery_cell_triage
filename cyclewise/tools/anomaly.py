"""Safety scan over early data: hot cells, IR jumps, missing sensors.

Flags are routed to the human approval step. They are not features unless the
Evidence agent proposes that and it is logged.
"""

from __future__ import annotations

import numpy as np

from cyclewise.config import load_config
from cyclewise.tools.cutoff_view import list_cells, query_early


def scan(batch: str) -> list[dict]:
    cfg = load_config()["safety"]
    ws = load_config()["preregistered"]["window_start"]
    cells = list_cells(batch)
    df = query_early(batch=batch, signals=["tmax", "ir", "glitch"], min_cycle=ws)
    flags: list[dict] = []
    per = {}
    for cid in cells["cell_id"]:
        g = df[(df["cell_id"] == cid) & ~df["glitch"].astype(bool)]
        t = g["tmax"].replace(0, np.nan)
        ir = g["ir"].replace(0, np.nan)
        if t.notna().sum() < 5:
            flags.append({"cell_id": cid, "flag": "sensor_missing", "detail": "temperature trace missing"})
            continue
        if ir.notna().sum() < 5:
            flags.append({"cell_id": cid, "flag": "sensor_missing", "detail": "IR trace missing"})
        per[cid] = {"tmax_mean": float(t.mean()),
                    "ir_jump": float(ir.dropna().diff().abs().max()) if ir.notna().sum() > 1 else 0.0}
    if per:
        tm = np.array([v["tmax_mean"] for v in per.values()])
        mu, sd = tm.mean(), tm.std(ddof=1) if len(tm) > 1 else 0.0
        for cid, v in per.items():
            z = (v["tmax_mean"] - mu) / sd if sd > 0 else 0.0
            if z > cfg["temp_z_threshold"]:
                flags.append({"cell_id": cid, "flag": "hot_cell",
                              "detail": f"early mean Tmax z={z:.2f} (> {cfg['temp_z_threshold']})"})
            if v["ir_jump"] > cfg["ir_jump_threshold_ohm"]:
                flags.append({"cell_id": cid, "flag": "ir_jump",
                              "detail": f"max IR step {v['ir_jump'] * 1000:.2f} mOhm "
                                        f"(> {cfg['ir_jump_threshold_ohm'] * 1000:.1f})"})
    return sorted(flags, key=lambda f: (f["cell_id"], f["flag"]))
