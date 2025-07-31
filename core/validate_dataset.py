import os
from typing import Tuple
import numpy as np


def validate_npz_dataset(path: str, use_intentions: bool = False) -> Tuple[bool, str]:
    """Simple validation for an .npz dataset.

    Parameters
    ----------
    path: str
        Path to the .npz file.
    use_intentions: bool
        Whether intention labels are expected.

    Returns
    -------
    Tuple[bool, str]
        ``(True, message)`` if dataset seems valid, ``(False, reason)`` otherwise.
    """
    if not path or not os.path.isfile(path):
        return False, f"File not found: {path}"
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:  # pragma: no cover - fail to load
        return False, f"Failed to load dataset: {exc}"

    required = {"observations", "actions"}
    missing = required - set(data.files)
    if missing:
        return False, f"Missing arrays: {', '.join(sorted(missing))}"

    obs = data["observations"]
    acts = data["actions"]
    if len(obs) != len(acts):
        return False, "Observations and actions length mismatch"

    if use_intentions:
        if "intentions" in data:
            intents = data["intentions"]
        elif "intentions_high" in data:
            intents = data["intentions_high"]
        else:
            return False, "Intentions array missing"
        if len(intents) != len(obs):
            return False, "Intentions length mismatch"

    return True, f"{len(obs)} samples"
