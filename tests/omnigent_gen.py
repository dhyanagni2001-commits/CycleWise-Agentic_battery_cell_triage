import importlib.util
from pathlib import Path

_p = Path(__file__).resolve().parent.parent / "omnigent" / "gen_tools.py"
_spec = importlib.util.spec_from_file_location("gen_tools", _p)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
AGENT_TOOLS = _m.AGENT_TOOLS
