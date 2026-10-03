"""Upload the local warehouse to Delta tables in Unity Catalog.

UNTESTED in this repo (no workspace was available during the build).
Requires: pip install -e ".[databricks]" and DATABRICKS_HOST / DATABRICKS_TOKEN
(or a ~/.databrickscfg profile) plus DATABRICKS_WAREHOUSE_ID for a SQL warehouse.

  python databricks/upload_tables.py --catalog cyclewise

Parquet files come from `python -m cyclewise.data.splits`
(warehouse/staging/delta_*.parquet). Files are uploaded to a UC volume and
loaded with CREATE TABLE ... AS SELECT from read_files().
"""

import argparse
import os
from pathlib import Path

from databricks.sdk import WorkspaceClient

ROOT = Path(__file__).resolve().parent.parent
STAGING = ROOT / "warehouse" / "staging"
TABLES = {  # local parquet -> schema.table
    "delta_cells.parquet": "lab.cells",
    "delta_features_early.parquet": "lab.features_early",
    "delta_cycles_raw.parquet": "hidden.cycles_raw",
    "delta_labels_hidden.parquet": "hidden.labels_hidden",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="cyclewise")
    ap.add_argument("--volume", default="lab.staging")
    a = ap.parse_args()
    w = WorkspaceClient()
    wh = os.environ["DATABRICKS_WAREHOUSE_ID"]

    def sql(q: str) -> None:
        r = w.statement_execution.execute_statement(statement=q, warehouse_id=wh, wait_timeout="50s")
        print(r.status.state, q.splitlines()[0][:90])

    schema, vol = a.volume.split(".")
    sql(f"CREATE SCHEMA IF NOT EXISTS {a.catalog}.lab")
    sql(f"CREATE SCHEMA IF NOT EXISTS {a.catalog}.hidden")
    sql(f"CREATE VOLUME IF NOT EXISTS {a.catalog}.{schema}.{vol}")
    for fname, table in TABLES.items():
        dest = f"/Volumes/{a.catalog}/{schema}/{vol}/{fname}"
        with open(STAGING / fname, "rb") as fh:
            w.files.upload(dest, fh, overwrite=True)
        sql(f"CREATE OR REPLACE TABLE {a.catalog}.{table} AS SELECT * FROM read_files('{dest}', format => 'parquet')")
    # Labels must not sit in an agent-readable volume after loading.
    w.files.delete(f"/Volumes/{a.catalog}/{schema}/{vol}/delta_labels_hidden.parquet")
    w.files.delete(f"/Volumes/{a.catalog}/{schema}/{vol}/delta_cycles_raw.parquet")
    sql(f"CREATE TABLE IF NOT EXISTS {a.catalog}.lab.research_log (seq BIGINT, ts STRING, run_id STRING, "
        "batch STRING, agent STRING, event STRING, input_hash STRING, output_json STRING, model STRING, "
        "prompt_version STRING, fallback_used BOOLEAN, prev_hash STRING, row_hash STRING)")
    print("Now run databricks/01_unity_catalog.sql to create the cycles_early view and grants.")


if __name__ == "__main__":
    main()
