import os

import pytest

os.environ.setdefault("CYCLEWISE_MLFLOW", "off")
os.environ.setdefault("CYCLEWISE_LLM", "offline")


@pytest.fixture
def tmp_log(tmp_path, monkeypatch):
    """Point the research log at a throwaway DB so tests never touch the real one."""
    from cyclewise.config import load_config
    cfg = load_config()
    monkeypatch.setitem(cfg["storage"], "log_db", str(tmp_path / "log.duckdb"))
    return tmp_path
