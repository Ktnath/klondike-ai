import importlib
import logging
import sys

REQUIRED_MODULES = [
    "klondike_core",
    "torch",
    "yaml",
    "gymnasium",  # migrated from gym to gymnasium
]

logging.basicConfig(level=logging.INFO)
missing = []
for mod in REQUIRED_MODULES:
    try:
        importlib.import_module(mod)
    except Exception as exc:  # pragma: no cover - import check
        logging.error("Module '%s' is missing: %s", mod, exc)
        missing.append(mod)

if missing:
    sys.exit(1)

logging.info("\u2705 Environnement prêt")

