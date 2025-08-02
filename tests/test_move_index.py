import importlib.util
from pathlib import Path
import sysconfig


def test_move_index_roundtrip():
    root = Path(__file__).resolve().parents[1]
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    so_path = root / ".venv" / "lib" / f"python{sysconfig.get_python_version()}" / "site-packages" / "klondike_core" / f"klondike_core{suffix}"
    spec = importlib.util.spec_from_file_location("klondike_core", so_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    idx = module.move_index("DS 0")
    assert idx != -1
    assert module.move_from_index(idx) == "DS 0"
