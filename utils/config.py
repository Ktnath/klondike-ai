"""Configuration loading utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import logging
import yaml


class DotDict(dict):
    """Dictionary with dot-access to nested keys."""

    def __getattr__(self, key: str) -> Any:  # noqa: D401
        try:
            value = self[key]
        except KeyError as exc:  # pragma: no cover - error path
            raise AttributeError(key) from exc
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
            self[key] = value
        return value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_config(path: str = "config.yaml") -> DotDict:
    """Load YAML configuration file.

    Parameters
    ----------
    path:
        Path to the configuration YAML file.

    Returns
    -------
    DotDict
        Configuration data accessible by keys or attributes.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with config_path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}

    if "training" in data and "learning_rate" not in data.get("training", {}):
        logging.warning("Missing 'learning_rate' in training config")

    return DotDict(data)
