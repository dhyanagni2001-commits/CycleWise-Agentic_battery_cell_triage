"""v2 warehouse: sequential checkpoints at 50/100/150.

  warehouse/v2/screen.duckdb   agent-visible base: cells, cycles <= 50, Qdlin 2..50. No labels.
  warehouse/v2/hidden.duckdb   cycles_raw (all), qdlin_ckpt (Qdlin at 100/150 +-2), labels.
                               Only the runner's materialise step and the scorer read it.

Per run, cyclewise.v2.visible copies screen.duckdb and then adds only the
cycles the lab has paid for (approved extensions).

Batch 3 is loaded only with --with-b3, and only after the v2 pre-registration
verifies (committed, hash-locked).

Run:  python -m cyclewise.v2.data [--with-b3]
"""

from __future__ import annotations

import argparse
import json

import duckdb
import numpy as np
import pandas as pd

from cyclewise import prereg
from cyclewise.config import load_config, path
from cyclewise.data.download import fetch
from cyclewise.data.load_raw import assert_ranges, clean_summary, compute_label, merge_continuations, read_batch

V2 = "warehouse/v2"
SCREEN = f"{V2}/screen.duckdb"
HIDDEN = f"{V2}/hidden.duckdb"
CKPT_TOL = 2   # use the nearest recorded Qdlin within +-2 cycles of a checkpoint


def batch_files(with_b3: bool) -> dict[str, dict]:
    v1 = load_config()["batches"]
    pre = prereg.load()["batches"]
    out = {k: {"file": v1[k]["file"], "url": v1[k].get("url"), "bytes": v1[k].get("bytes"),
               "role": pre[k]["role"], "exclude": []} for k in ("b1", "b2")}
    if with_b3:
        b3 = pre["b3"]
        out["b3"] = {"file": b3["file"], "url": b3["url"], "bytes": b3["bytes"], "role": b3["role"],
                     "exclude": b3["exclude"]}
    return out


def within_batch_labels(lab: pd.DataFrame) -> pd.DataFrame:
    """Primary v2 label: >= the batch's own Q75 of cycle life (censored = lower bound)."""
    q = prereg.load()["labels"]["quantile"]
    out = []
    for _, g in lab.groupby("batch"):
        g = g.copy()
        thr = float(np.quantile(g["cycle_life"].astype(float), q))
        g["threshold"] = thr
        g["label_known"] = ~g["censored"] | (g["cycle_life"] >= thr)
        g["long_lived"] = np.where(g["label_known"], g["cycle_life"] >= thr, None)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def build(with_b3: bool = False) -> dict:
    cfg = load_config()
    meta = prereg.verify()                      # refuses if the pre-registration changed / is uncommitted
    files = batch_files(with_b3)
    if with_b3 and not path(files["b3"]["file"]).exists():
        print("downloading batch 3 (pre-registration verified:", meta["prereg_commit"][:8], ")")
        fetch(files["b3"]["url"], path(files["b3"]["file"]), files["b3"]["bytes"])

    raw, grids = {}, {}
    for key, b in files.items():
        raw[key], grids[key] = read_batch(path(b["file"]), key, early_cutoff=150)
    merge_continuations(raw["b1"], raw["b2"], cfg["continuations"])
    vgrid = grids["b1"]
    for k, g in grids.items():
        assert np.allclose(g, vgrid), f"voltage grid differs in {k}"

    cells, cycles, q_early, q_ckpt, labels = [], [], [], [], []
    for key, bcells in raw.items():
        for cid, c in bcells.items():
            if cid in files[key]["exclude"]:
                cells.append({"cell_id": cid, "batch": key, "protocol": c.protocol, "status": "excluded",
                              "status_reason": "noisy channel (authors' reference loader)"})
                continue
            df, qc = clean_summary(c.summary, cfg)
            warn = assert_ranges(df, cfg, cid)
            status = "eligible"
            if 10 not in c.qdlin or 50 not in c.qdlin:
                status = "insufficient_early_data"
            cells.append({"cell_id": cid, "batch": key, "protocol": c.protocol, "status": status,
                          "status_reason": "" if status == "eligible" else "Qdlin missing at 10 or 50",
                          "notes": "; ".join(c.notes + warn), **{f"qc_{k}": v for k, v in qc.items()}})
            df.insert(0, "cell_id", cid)
            df.insert(1, "batch", key)
            cycles.append(df)
            for cyc, q in c.qdlin.items():
                row = {"cell_id": cid, "batch": key, "cycle": int(cyc), "qdlin": q.astype(np.float32)}
                if 2 <= cyc <= 50:
                    q_early.append(row)
                elif any(abs(cyc - ck) <= CKPT_TOL for ck in (100, 150)):
                    q_ckpt.append(row)
            labels.append({"cell_id": cid, "batch": key,
                           **compute_label(df, cfg["preregistered"]["eol_capacity_ah"],
                                           cfg["cleaning"]["eol_stop_tol_ah"]),
                           "file_cycle_life": c.file_cycle_life})

    cells_df = pd.DataFrame(cells)
    cyc_df = pd.concat(cycles, ignore_index=True)
    cyc_df["cycle"] = cyc_df["cycle"].astype(int)
    lab_df = within_batch_labels(pd.DataFrame(labels))
    roles = {k: b["role"] for k, b in files.items()}

    def lists(rows):
        d = pd.DataFrame(rows)
        d["qdlin"] = d["qdlin"].apply(lambda a: [float(x) for x in a])
        return d

    for rel in (SCREEN, HIDDEN):
        p = path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            p.unlink()
    with duckdb.connect(str(path(SCREEN))) as con:
        con.register("c", cells_df); con.execute("CREATE TABLE cells AS SELECT * FROM c")
        con.register("y", cyc_df[cyc_df["cycle"] <= 50]); con.execute("CREATE TABLE cycles AS SELECT * FROM y")
        con.register("q", lists(q_early)); con.execute("CREATE TABLE qdlin AS SELECT * FROM q")
        con.register("v", pd.DataFrame({"i": np.arange(vgrid.size), "voltage": vgrid}))
        con.execute("CREATE TABLE voltage_grid AS SELECT * FROM v")
        con.execute("CREATE TABLE roles (batch VARCHAR, role VARCHAR)")
        con.executemany("INSERT INTO roles VALUES (?, ?)", list(roles.items()))
        assert con.execute("SELECT max(cycle) FROM cycles").fetchone()[0] <= 50
        assert con.execute("SELECT max(cycle) FROM qdlin").fetchone()[0] <= 50
    with duckdb.connect(str(path(HIDDEN))) as con:
        con.register("r", cyc_df); con.execute("CREATE TABLE cycles_raw AS SELECT * FROM r")
        con.register("k", lists(q_ckpt)); con.execute("CREATE TABLE qdlin_ckpt AS SELECT * FROM k")
        lab = lab_df.copy(); lab["long_lived"] = lab["long_lived"].astype("object")
        con.register("l", lab); con.execute("CREATE TABLE labels AS SELECT * FROM l")
        con.execute("CREATE TABLE meta (k VARCHAR, v VARCHAR)")
        con.execute("INSERT INTO meta VALUES ('prereg', ?)", [json.dumps(meta)])

    counts = {b: {"eligible": int(((cells_df["batch"] == b) & (cells_df["status"] == "eligible")).sum()),
                  "excluded": int(((cells_df["batch"] == b) & (cells_df["status"] == "excluded")).sum()),
                  "censored": int(lab_df[lab_df["batch"] == b]["censored"].sum()),
                  "label_unknown": int((~lab_df[lab_df["batch"] == b]["label_known"]).sum())}
              for b in files}
    return {"prereg": meta, "counts": counts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-b3", action="store_true", help="include batch 3 (test); requires the committed v2 pre-registration")
    a = ap.parse_args()
    out = build(a.with_b3)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
