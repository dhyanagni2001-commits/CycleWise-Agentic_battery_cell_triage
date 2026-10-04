"""Databricks upload script: offline checks (no workspace needed)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("upload_tables", ROOT / "databricks" / "upload_tables.py")
up = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(up)


def test_uc_sql_parses_cleanly_without_principal():
    stmts = up.uc_statements("workspace", None)
    assert not any("GRANT" in s for s in stmts)
    assert not any(s.upper().startswith("CREATE CATALOG") for s in stmts)
    assert any("CREATE OR REPLACE VIEW workspace.lab.cycles_early" in s and "cycle <= 50" in s for s in stmts)
    assert all(s.split()[0].upper() in {"CREATE", "GRANT"} for s in stmts)   # no comment fragments


def test_uc_sql_grants_only_lab_to_agent():
    stmts = up.uc_statements("workspace", "agents@example.com")
    grants = [s for s in stmts if s.startswith("GRANT")]
    assert grants and all("`agents@example.com`" in s for s in grants)
    assert not any("hidden" in s for s in grants)          # the agent gets nothing in hidden.*
    assert not any("${" in s for s in stmts)              # every placeholder filled
