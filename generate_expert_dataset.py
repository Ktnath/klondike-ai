import argparse
import csv
import os
from collections import Counter
from typing import List, Tuple

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


def generate_games(
    num_games: int,
    output: str,
    use_intentions: bool = False,
    to_csv: str | None = None,
    csv_only: bool = False,
) -> None:
    """Generate a dataset of optimal moves using the Rust solver."""

    if not csv_only:
        os.makedirs(os.path.dirname(output), exist_ok=True)

    dataset: List[Tuple[np.ndarray, int, str]] = []

    for _ in trange(num_games, desc="games"):
        state = new_game(str(shuffle_seed()))
        obs = encode_observation(state)
        solution = solve_klondike(state)

        for mv, intention in solution:
            obs_arr = np.array(obs, dtype=np.float32)
            idx = int(move_index(mv))
            if idx == -1:
                print(f"⚠️ move_index invalide pour le move : {mv}")
            dataset.append((obs_arr, idx, intention if use_intentions else ""))
            state, _ = play_move(state, mv)
            obs = encode_observation(state)

    if not dataset:
        raise ValueError("Dataset is empty")

    observations = [row[0] for row in dataset]
    actions = [row[1] for row in dataset]
    intentions = [row[2] for row in dataset]

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

    if to_csv:
        os.makedirs(os.path.dirname(to_csv), exist_ok=True)
        obs_len = len(dataset[0][0])
        with open(to_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["move_index", "intention"] + [f"obs_{i}" for i in range(obs_len)])
            for obs_row, move_idx, intent in dataset:
                writer.writerow([move_idx, intent] + list(obs_row))

        size = os.path.getsize(to_csv)
        with open(to_csv, "r", newline="") as f:
            line_count = sum(1 for _ in f) - 1  # minus header
        print(f"CSV exporté vers {to_csv} ({line_count} lignes, {size} octets)")

    if not csv_only:
        np.savez_compressed(output, observations=obs_array, actions=actions_array, intentions=intentions_array)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate expert dataset")
    parser.add_argument("--games", type=int, default=1000, help="Number of games")
    parser.add_argument("--output", type=str, default="data/expert_dataset.npz", help="Output path")
    parser.add_argument("--use_intentions", action="store_true", help="Use solver intentions")
    parser.add_argument("--to_csv", type=str, help="Optional path to export CSV in addition to .npz")
    parser.add_argument("--csv_only", action="store_true", help="Export only to CSV, skip .npz")
    args = parser.parse_args()

    if args.csv_only and not args.to_csv:
        parser.error("--csv_only requires --to_csv path")

    generate_games(args.games, args.output, args.use_intentions, args.to_csv, args.csv_only)


if __name__ == "__main__":
    main()

