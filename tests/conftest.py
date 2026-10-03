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


def _warehouse_ready() -> bool:
    from cyclewise.config import load_config, path
    st = load_config()["storage"]
    return path(st["early_db"]).exists() and path(st["hidden_db"]).exists()


def pytest_collection_modifyitems(config, items):
    """Tests marked requires_data need the built warehouse (5 GB raw data, not in git).
    In a fresh clone they are skipped with instructions instead of failing."""
    if _warehouse_ready():
        return
    skip = pytest.mark.skip(reason="warehouse not built: run `python -m cyclewise.data.download`, "
                                   "`python -m cyclewise.data.load_raw`, `python -m cyclewise.data.splits`")
    for item in items:
        if "requires_data" in item.keywords:
            item.add_marker(skip)
