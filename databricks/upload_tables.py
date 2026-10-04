"""Load CycleWise tables into Databricks Unity Catalog and apply the agent permissions.

What it does (in order):
  1. creates schemas  <catalog>.lab   (agent-visible)  and  <catalog>.hidden  (labels, full traces)
  2. uploads Parquet to a UC volume in the SAME schema as each table (hidden data never sits in lab)
  3. creates Delta tables: lab.cells, lab.features_early, lab.research_log,
                           hidden.cycles_raw, hidden.labels_hidden
  4. runs databricks/01_unity_catalog.sql: the lab.cycles_early view (cycles <= 50) and, with
     --agent-principal, the GRANTs that keep the agent out of hidden.*
  5. deletes the staged files from the volumes

Auth: any Databricks SDK method, e.g. `databricks auth login --host https://<workspace>` (writes
~/.databrickscfg), or DATABRICKS_HOST + DATABRICKS_TOKEN. SQL runs on a SQL warehouse:
--warehouse-id or DATABRICKS_WAREHOUSE_ID, else the first warehouse found.

  python databricks/upload_tables.py --dry-run                      # check files + print every statement
  python databricks/upload_tables.py --catalog workspace            # Free Edition default catalog
  python databricks/upload_tables.py --catalog workspace --agent-principal agents@example.com
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
STAGING = ROOT / "warehouse" / "staging"
LOG_DB = ROOT / "warehouse" / "research_log.duckdb"
UC_SQL = Path(__file__).resolve().parent / "01_unity_catalog.sql"
TABLES = {  # local parquet -> schema.table
    "delta_cells.parquet": "lab.cells",
    "delta_features_early.parquet": "lab.features_early",
    "research_log.parquet": "lab.research_log",
    "delta_cycles_raw.parquet": "hidden.cycles_raw",
    "delta_labels_hidden.parquet": "hidden.labels_hidden",
}


def export_research_log() -> Path:
    out = STAGING / "research_log.parquet"
    with duckdb.connect(str(LOG_DB), read_only=True) as con:
        con.execute(f"COPY (SELECT * FROM research_log ORDER BY seq) TO '{out}' (FORMAT parquet)")
    return out


def uc_statements(catalog: str, principal: str | None) -> list[str]:
    """01_unity_catalog.sql with placeholders filled. GRANTs only when a principal is given."""
    text = UC_SQL.read_text().replace("${catalog}", catalog)
    # Drop comments FIRST (they may contain ';'), then split into statements.
    code = "\n".join(ln.split("--", 1)[0] for ln in text.splitlines())
    stmts = []
    for raw in code.split(";"):
        s = " ".join(raw.split())
        if not s:
            continue
        if "${agent_principal}" in s or "${scorer_principal}" in s:
            if not principal:
                continue
            s = s.replace("${agent_principal}", principal).replace("${scorer_principal}", principal + "_scorer")
            if "_scorer`" in s:   # scorer grants are optional; the uploading user already owns the tables
                continue
        if s.upper().startswith("CREATE CATALOG"):
            continue          # the catalog must already exist (Free Edition: `workspace`)
        stmts.append(s)
    return stmts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="workspace", help="existing UC catalog (Free Edition: workspace)")
    ap.add_argument("--warehouse-id", default=os.environ.get("DATABRICKS_WAREHOUSE_ID"))
    ap.add_argument("--agent-principal", help="user/group/service principal the agents run as; enables GRANTs")
    ap.add_argument("--dry-run", action="store_true", help="check local files and print statements; no network")
    a = ap.parse_args()

    missing = [f for f in TABLES if f != "research_log.parquet" and not (STAGING / f).exists()]
    if missing or not LOG_DB.exists():
        sys.exit(f"missing local data {missing or [str(LOG_DB)]}; run `python -m cyclewise.data.splits` "
                 "and at least one study run first")
    export_research_log()

    plan: list[tuple[str, str]] = [("sql", f"CREATE SCHEMA IF NOT EXISTS {a.catalog}.{s}") for s in ("lab", "hidden")]
    plan += [("sql", f"CREATE VOLUME IF NOT EXISTS {a.catalog}.{s}.staging") for s in ("lab", "hidden")]
    for fname, table in TABLES.items():
        schema = table.split(".")[0]
        dest = f"/Volumes/{a.catalog}/{schema}/staging/{fname}"
        plan.append(("upload", f"{STAGING / fname} -> {dest}"))
        plan.append(("sql", f"CREATE OR REPLACE TABLE {a.catalog}.{table} AS "
                            f"SELECT * FROM read_files('{dest}', format => 'parquet')"))
    plan += [("sql", s) for s in uc_statements(a.catalog, a.agent_principal)]
    plan += [("delete", f"/Volumes/{a.catalog}/{t.split('.')[0]}/staging/{f}") for f, t in TABLES.items()]
    plan.append(("check", f"SELECT max(cycle) AS max_cycle FROM {a.catalog}.lab.cycles_early"))

    if a.dry_run:
        for kind, what in plan:
            print(f"[{kind}] {what}")
        print(f"\n{len(plan)} steps. Local files OK. Nothing was sent.")
        return

    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.sql import StatementState

    w = WorkspaceClient()
    wh = a.warehouse_id or next((x.id for x in w.warehouses.list()), None)
    if not wh:
        sys.exit("no SQL warehouse found; create one (SQL Warehouses -> Create) or pass --warehouse-id")
    print(f"workspace {w.config.host}  warehouse {wh}  catalog {a.catalog}")

    def run_sql(stmt: str):
        r = w.statement_execution.execute_statement(statement=stmt, warehouse_id=wh, wait_timeout="50s")
        while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
            time.sleep(2)
            r = w.statement_execution.get_statement(r.statement_id)
        if r.status.state != StatementState.SUCCEEDED:
            err = r.status.error.message if r.status.error else r.status.state
            sys.exit(f"FAILED: {stmt.splitlines()[0][:100]}\n  {err}")
        print(f"ok   {stmt.splitlines()[0][:100]}")
        return r

    for kind, what in plan:
        if kind == "sql":
            run_sql(what)
        elif kind == "check":
            r = run_sql(what)
            mx = int(r.result.data_array[0][0]) if r.result and r.result.data_array else None
            print(f"     cycles_early max cycle = {mx}  ({'OK: cutoff holds' if mx is not None and mx <= 50 else 'PROBLEM'})")
        elif kind == "upload":
            src, dest = what.split(" -> ")
            with open(src, "rb") as fh:
                w.files.upload(dest, fh, overwrite=True)
            print(f"up   {dest}")
        else:
            w.files.delete(what)
            print(f"del  {what}")
    print("\nDone. Tables are in", f"{a.catalog}.lab and {a.catalog}.hidden.",
          "Next: run databricks/02_dashboard_queries.sql in the SQL editor (replace ${catalog}).")


if __name__ == "__main__":
    main()
