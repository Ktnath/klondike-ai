import os
from typing import Tuple
import numpy as np

_MESSAGES_SUCCESS = {
    False: "✅ Dataset valide sans intentions",
    True: "✅ Dataset valide avec intentions",
}

_MESSAGES_FAILURE = {
    "file": "❌ Fichier introuvable",
    "load": "❌ Échec de lecture du fichier",
    "missing_fields": "❌ Champs obligatoires manquants (observations/actions)",
    "obs_shape": "❌ Shape des observations invalide",
    "act_shape": "❌ Shape des actions invalide",
    "int_shape": "❌ Shape des intentions invalide",
    "int_missing": "❌ Dataset sans intentions alors que requis",
    "degenerate": "❌ Actions dégénérées (toutes identiques)",
}

def validate_npz_dataset(path: str, use_intentions: bool = False, verbose: bool = True) -> Tuple[bool, str]:
    if not path or not os.path.isfile(path):
        return False, _MESSAGES_FAILURE["file"]

    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        return False, f"{_MESSAGES_FAILURE['load']}: {e}"

    if "observations" not in data or "actions" not in data:
        return False, _MESSAGES_FAILURE["missing_fields"]

    obs = data["observations"]
    actions = data["actions"]
    intents = data.get("intentions", None)

    if obs.ndim != 2 or obs.shape[1] not in (156, 160):
        return False, _MESSAGES_FAILURE["obs_shape"]
    if actions.ndim != 1 or actions.shape[0] != obs.shape[0]:
        return False, _MESSAGES_FAILURE["act_shape"]

    if use_intentions:
        if intents is None:
            return False, _MESSAGES_FAILURE["int_missing"]
        if intents.ndim != 1 or intents.shape[0] != obs.shape[0]:
            return False, _MESSAGES_FAILURE["int_shape"]

    if actions.size > 0 and np.all(actions == actions[0]):
        return False, _MESSAGES_FAILURE["degenerate"]

    if verbose:
        print("🔎 Distribution des actions :")
        uniq, counts = np.unique(actions, return_counts=True)
        for u, c in zip(uniq, counts):
            print(f"  Action {u}: {c} exemples")
        if intents is not None:
            print("🧠 Intentions (premières 10):", intents[:10])

    return True, _MESSAGES_SUCCESS[use_intentions]
