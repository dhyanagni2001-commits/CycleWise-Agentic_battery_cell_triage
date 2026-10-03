"""Early-cycle features (cycles window_start..early_cutoff only).

Every feature here is computed from data with cycle_index <= 50. The input
frames are read from the staging tables and filtered to the early window
before any computation, and the function asserts that.

Feature catalog (name -> description, prior direction for long life):
  dq_logvar     log10 var(ΔQ_{50-10}(V))                    (-)  Severson variance feature
  dq_logmin     log10 |min ΔQ_{50-10}(V)|                   (-)
  dq_logskew    log10 |skewness ΔQ_{50-10}(V)|              (?)
  dq_logkurt    log10 |kurtosis ΔQ_{50-10}(V)|              (?)
  q2            discharge capacity at cycle 2 (Ah)          (+)
  qmax_minus_q2 max(Q, cycles 2..50) - Q(cycle 2) (Ah)      (+)
  q50           mean discharge capacity, cycles 46..50 (Ah) (+)  early-capacity baseline
  fade_slope    slope of Q vs cycle, cycles 2..50 (Ah/cyc)  (+)
  fade_slope_late slope of Q vs cycle, cycles 40..50        (+)
  fade_intercept intercept of that fit                      (?)
  chargetime_2_6 mean charge time, cycles 2..6 (min)        (+)
  ir_2          internal resistance at cycle 2 (ohm)        (-)
  ir_min        min IR, cycles 2..50                        (-)
  ir_delta      IR(cycle 50) - IR(cycle 2)                  (-)
  tmax_mean     mean max temperature, cycles 2..50 (C)      (-)
  tavg_int      sum of mean temperature, cycles 2..50       (-)
  c1, soc1, c2  charging protocol parameters (known before testing)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats

FEATURE_CATALOG: dict[str, dict] = {
    "dq_logvar": {"desc": "log10 var of ΔQ_{50-10}(V)", "prior_sign": -1},
    "dq_logmin": {"desc": "log10 |min ΔQ_{50-10}(V)|", "prior_sign": -1},
    "dq_logskew": {"desc": "log10 |skewness ΔQ_{50-10}(V)|", "prior_sign": 0},
    "dq_logkurt": {"desc": "log10 |kurtosis ΔQ_{50-10}(V)|", "prior_sign": 0},
    "q2": {"desc": "discharge capacity at cycle 2 (Ah)", "prior_sign": 1},
    "qmax_minus_q2": {"desc": "max(Q, cycles 2..50) - Q(2) (Ah)", "prior_sign": 1},
    "q50": {"desc": "mean discharge capacity, cycles 46..50 (Ah)", "prior_sign": 1},
    "fade_slope": {"desc": "slope of Q vs cycle, cycles 2..50 (Ah/cycle)", "prior_sign": 1},
    "fade_slope_late": {"desc": "slope of Q vs cycle, cycles 40..50 (Ah/cycle)", "prior_sign": 1},
    "fade_intercept": {"desc": "intercept of the 2..50 linear fit (Ah)", "prior_sign": 0},
    "chargetime_2_6": {"desc": "mean charge time, cycles 2..6 (min)", "prior_sign": 1},
    "ir_2": {"desc": "internal resistance at cycle 2 (ohm)", "prior_sign": -1},
    "ir_min": {"desc": "min internal resistance, cycles 2..50 (ohm)", "prior_sign": -1},
    "ir_delta": {"desc": "IR(50) - IR(2) (ohm)", "prior_sign": -1},
    "tmax_mean": {"desc": "mean max temperature, cycles 2..50 (C)", "prior_sign": -1},
    "tavg_int": {"desc": "sum of mean temperature, cycles 2..50 (C*cycle)", "prior_sign": -1},
    "c1": {"desc": "first-step charge C-rate (protocol)", "prior_sign": -1},
    "soc1": {"desc": "SOC % at which the charge rate switches (protocol)", "prior_sign": 0},
    "c2": {"desc": "second-step charge C-rate (protocol)", "prior_sign": -1},
}

_PROTO = re.compile(r"([\d.]+)C\(([\d.]+)%\)-([\d.]+)C?")


def parse_protocol(p: str) -> dict:
    """'3.6C(80%)-3.6C' -> c1=3.6, soc1=80, c2=3.6. Unparseable protocols give NaN."""
    m = _PROTO.search(p or "")
    if not m:
        return {"c1": np.nan, "soc1": np.nan, "c2": np.nan}
    return {"c1": float(m.group(1)), "soc1": float(m.group(2)), "c2": float(m.group(3))}


def _slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3:
        return np.nan, np.nan
    b, a = np.polyfit(x, y, 1)
    return float(b), float(a)


def _safe_log10_abs(v: float) -> float:
    v = abs(v)
    return float(np.log10(v)) if v > 0 and np.isfinite(v) else np.nan


def delta_q(q_hi: np.ndarray, q_lo: np.ndarray) -> np.ndarray:
    """ΔQ(V) = Q_hi(V) - Q_lo(V) on the shared voltage grid; non-finite points dropped."""
    d = q_hi - q_lo
    return d[np.isfinite(d)]


def cell_features(cyc: pd.DataFrame, qd: dict[int, np.ndarray], protocol: str,
                  window_start: int, cutoff: int, dq_cycles: tuple[int, int]) -> dict:
    assert cyc["cycle"].max() <= cutoff, "featurize received data past the early cutoff"
    w = cyc[(cyc["cycle"] >= window_start) & ~cyc["glitch"]].sort_values("cycle")
    f: dict[str, float] = {}
    lo, hi = dq_cycles
    if lo in qd and hi in qd:
        d = delta_q(qd[hi], qd[lo])
        f["dq_logvar"] = _safe_log10_abs(np.var(d))
        f["dq_logmin"] = _safe_log10_abs(np.min(d))
        f["dq_logskew"] = _safe_log10_abs(stats.skew(d))
        f["dq_logkurt"] = _safe_log10_abs(stats.kurtosis(d))
    q = w.set_index("cycle")["qd"]
    f["q2"] = float(q.loc[window_start]) if window_start in q.index else float(q.iloc[0])
    f["qmax_minus_q2"] = float(q.max() - f["q2"])
    f["q50"] = float(q[q.index >= cutoff - 4].mean())
    f["fade_slope"], f["fade_intercept"] = _slope(q.index.values.astype(float), q.values)
    late = q[q.index >= 40]
    f["fade_slope_late"], _ = _slope(late.index.values.astype(float), late.values)
    ct = w[w["cycle"] <= window_start + 4]["chargetime"]
    f["chargetime_2_6"] = float(ct.mean())
    ir = w.set_index("cycle")["ir"].replace(0, np.nan)
    f["ir_2"] = float(ir.dropna().iloc[0]) if ir.notna().any() else np.nan
    f["ir_min"] = float(ir.min())
    f["ir_delta"] = float(ir.dropna().iloc[-1] - ir.dropna().iloc[0]) if ir.notna().sum() > 1 else np.nan
    f["tmax_mean"] = float(w["tmax"].mean())
    f["tavg_int"] = float(w["tavg"].sum())
    f.update(parse_protocol(protocol))
    return f


def build_features(cells: pd.DataFrame, cycles_early: pd.DataFrame, qdlin_early: pd.DataFrame,
                   cfg: dict) -> pd.DataFrame:
    pre = cfg["preregistered"]
    cutoff = pre["early_cutoff"]
    assert cycles_early["cycle"].max() <= cutoff
    assert qdlin_early["cycle"].max() <= cutoff
    by_cell_cyc = dict(tuple(cycles_early.groupby("cell_id")))
    by_cell_q = {cid: dict(zip(g["cycle"], g["qdlin"])) for cid, g in qdlin_early.groupby("cell_id")}
    rows = []
    for _, c in cells.iterrows():
        if c["status"] == "excluded":
            continue
        f = cell_features(by_cell_cyc[c["cell_id"]], by_cell_q.get(c["cell_id"], {}), c["protocol"],
                          pre["window_start"], cutoff, tuple(pre["delta_q_cycles"]))
        rows.append({"cell_id": c["cell_id"], "batch": c["batch"], **f})
    return pd.DataFrame(rows)
