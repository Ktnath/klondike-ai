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


def get_config_value(config: DotDict, path: str, default: Any | None = None) -> Any:
    """Safely retrieve a nested configuration value."""
    current: Any = config
    for part in path.split("."):
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            logging.warning("Config key '%s' missing, using default %r", path, default)
            return default
    return current


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


def get_input_dim(config: DotDict) -> int:
    """Return the observation vector dimension based on intentions usage.

    Parameters
    ----------
    config:
        Loaded configuration object.

    Returns
    -------
    int
        Size of the observation vector expected by the models.
    """
    base_dim = int(getattr(config.env, "observation_dim", 156))
    use_int = bool(getattr(config.env, "use_intentions", False))
    return base_dim + 4 if use_int else base_dim
