from __future__ import annotations

"""Dataset validation utilities."""

from typing import Tuple

import numpy as np


_DEF_SUCCESS = {
    False: "Dataset valide sans intentions",
    True: "Dataset valide avec intentions",
}


def validate_npz_dataset(path: str, use_intentions: bool = False, verbose: bool = True) -> Tuple[bool, str]:
    """Validate the structure and content of a ``.npz`` dataset."""
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:  # pragma: no cover - load failure
        return False, f"Erreur de lecture: {exc}"

    if "observations" not in data or "actions" not in data:
        return False, "champs manquants"

    obs = data["observations"]
    actions = data["actions"]
    intents = data["intentions"] if "intentions" in data else None

    if obs.ndim != 2 or obs.shape[1] not in (156, 160):
        return False, "shape observations invalide"
    if actions.ndim != 1 or actions.shape[0] != obs.shape[0]:
        return False, "shape actions invalide"

    if use_intentions:
        if intents is None:
            return False, "intentions manquantes"
        if intents.ndim != 1 or intents.shape[0] != obs.shape[0]:
            return False, "shape intentions invalide"

    if actions.size > 0 and np.all(actions == actions.flat[0]):
        return False, "actions dégénérées"

    if verbose:
        uniq, counts = np.unique(actions, return_counts=True)
        print("Distribution des actions:")
        for u, c in zip(uniq, counts):
            print(f"  action {u}: {c}")
        if intents is not None:
            print("Intentions (premières 10):", intents[:10])

    return True, _DEF_SUCCESS[bool(use_intentions)]
