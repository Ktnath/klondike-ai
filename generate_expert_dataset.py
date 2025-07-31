import argparse
import json
import os
import random
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

from env.klondike_env import KlondikeEnv


def _parse_solution(data: list | dict | None) -> List[str]:
    """Extract move strings from solver output."""
    if not data:
        return []
    if isinstance(data, list):
        moves = []
        for item in data:
            if isinstance(item, list) and item:
                moves.append(item[0])
            elif isinstance(item, dict):
                mv = item.get("move") or item.get("mv") or item.get("action")
                if mv:
                    moves.append(mv)
            else:
                moves.append(str(item))
        return moves
    result = data.get("result", [])
    if isinstance(result, list):
        return _parse_solution(result)
    return []


def generate_games(num_games: int, output: str, use_intentions: bool = False) -> None:
    """Generate a dataset of optimal moves using the Rust solver."""
    os.makedirs(os.path.dirname(output), exist_ok=True)

    observations: List[np.ndarray] = []
    actions: List[int] = []
    intentions: List[str] = []

    for _ in trange(num_games, desc="games"):
        seed = str(shuffle_seed()) if shuffle_seed else str(random.randint(0, 2**32 - 1))
        env = KlondikeEnv(seed=seed, use_intentions=use_intentions)
        obs, _ = env.reset(seed)

        solution = json.loads(solve_klondike(env.state))
        moves = _parse_solution(solution)

        for mv in moves:
            observations.append(np.array(obs, dtype=np.float32))
            act_idx = int(move_index(mv))
            actions.append(act_idx)

            obs, _, _, _, info = env.step(act_idx)
            intentions.append(info.get("intention", ""))

    obs_array = np.stack(observations)
    actions_array = np.array(actions, dtype=np.int64)
    intentions_array = np.array(intentions)

    if actions_array.size > 0 and np.all(actions_array == actions_array[0]):
        raise ValueError("Dataset contains no action diversity")

    np.savez_compressed(output, observations=obs_array, actions=actions_array, intentions=intentions_array)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate expert dataset")
    parser.add_argument("--games", type=int, default=1000, help="Number of games")
    parser.add_argument("--output", type=str, default="data/expert_dataset.npz", help="Output path")
    parser.add_argument("--use_intentions", action="store_true", help="Use environment intentions")
    args = parser.parse_args()

    generate_games(args.games, args.output, args.use_intentions)


if __name__ == "__main__":
    main()
