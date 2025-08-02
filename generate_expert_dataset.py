import argparse
import os
from collections import Counter
from typing import List

from tqdm import trange
import numpy as np

# Automatically patched for modular project structure via bootstrap.py
from bootstrap import *  # noqa: F401,F403

try:  # Prefer compiled extension from local crate
    from klondike_core import (
        new_game,
        play_move,
        move_index,
        solve_klondike,
        shuffle_seed,
        encode_observation,
    )
except Exception:  # pragma: no cover - fallback for old package name
    from core import new_game, play_move, move_index, solve_klondike, shuffle_seed, encode_observation


def generate_games(num_games: int, output: str, use_intentions: bool = False) -> None:
    """Generate a dataset of optimal moves using the Rust solver."""
    os.makedirs(os.path.dirname(output), exist_ok=True)

    observations: List[np.ndarray] = []
    actions: List[int] = []
    intentions: List[str] = []

    for _ in trange(num_games, desc="games"):
        state = new_game(str(shuffle_seed()))
        obs = encode_observation(state)
        solution = solve_klondike(state)

        for mv, intention in solution:
            observations.append(np.array(obs, dtype=np.float32))
            idx = int(move_index(mv))
            if idx == -1:
                print(f"⚠️ move_index invalide pour le move : {mv}")
            actions.append(idx)
            intentions.append(intention if use_intentions else "")
            state, _ = play_move(state, mv)
            obs = encode_observation(state)

    obs_array = np.stack(observations)
    actions_array = np.array(actions, dtype=np.int64)
    intentions_array = np.array(intentions)

    if actions_array.size > 0 and np.all(actions_array == actions_array[0]):
        print("\n🔍 Analyse de la diversité des actions et intentions :")
        print("Nombre d'actions uniques :", len(set(actions)))
        print("Nombre d'intentions uniques :", len(set(intentions)))
        print("Top actions :", Counter(actions).most_common(5))
        print("Top intentions :", Counter(intentions).most_common(5))
        raise ValueError("Dataset contains no action diversity")

    np.savez_compressed(output, observations=obs_array, actions=actions_array, intentions=intentions_array)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate expert dataset")
    parser.add_argument("--games", type=int, default=1000, help="Number of games")
    parser.add_argument("--output", type=str, default="data/expert_dataset.npz", help="Output path")
    parser.add_argument("--use_intentions", action="store_true", help="Use solver intentions")
    args = parser.parse_args()

    generate_games(args.games, args.output, args.use_intentions)


if __name__ == "__main__":
    main()

