import argparse
import json
import os
import random
from typing import List


from tqdm import trange
import numpy as np

# Automatically patched for modular project structure via bootstrap.py
from bootstrap import *

from klondike_core import (
    new_game,
    play_move,
    move_index,
    solve_klondike,
    shuffle_seed,
    encode_observation,
)


def generate_games(num_games: int, output: str, use_intentions: bool = False) -> None:
    """Generate expert dataset using the optimal solver.

    Parameters
    ----------
    num_games:
        Number of games to generate.
    output:
        Path to output ``.npz`` file.
    use_intentions:
        If ``True``, intention one-hot vectors are concatenated to each
        observation.
    """
    os.makedirs(os.path.dirname(output), exist_ok=True)

    def _intent_vec(label: str | None) -> List[float]:
        mapping = {
            "reveal": 0,
            "foundation": 1,
            "stack_move": 2,
            "king_to_empty": 3,
        }
        vec = [0.0, 0.0, 0.0, 0.0]
        if label:
            key = str(label).strip().lower().replace(" ", "_")
            for name, idx in mapping.items():
                if name in key:
                    vec[idx] = 1.0
                    break
        return vec

    observations: List[List[float]] = []
    actions: List[int] = []

    for _ in trange(num_games, desc="games"):
        seed = str(shuffle_seed()) if shuffle_seed else str(random.randint(0, 2**32 - 1))
        state = new_game(seed)
        solution = json.loads(solve_klondike(state))
        if isinstance(solution, list):
            result = [item[0] if isinstance(item, list) and item else item for item in solution]
            intents = [item[1] if isinstance(item, list) and len(item) > 1 else "" for item in solution]
        else:
            result = solution.get("result", [])
            intents = solution.get("intentions")
        prev_state = state

        for idx, item in enumerate(result):
            if isinstance(item, dict):
                mv_json = item.get("move") or item.get("mv") or item.get("action")
                intention = item.get("intention", "")
            else:
                mv_json = item
                if isinstance(intents, list) and idx < len(intents):
                    intention = intents[idx]
                else:
                    intention = ""

            obs_vector = np.array(encode_observation(prev_state), dtype=np.float32)
            action = move_index(mv_json)
            next_state, _ = play_move(prev_state, mv_json)

            if use_intentions:
                intention_vector = np.array(_intent_vec(intention), dtype=np.float32)
                obs_vector = np.concatenate([obs_vector, intention_vector])

            observations.append(obs_vector.tolist())
            actions.append(int(action))

            prev_state = next_state

    np.savez(
        output,
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.int64),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate expert dataset")
    parser.add_argument("--games", type=int, default=100, help="Number of games")
    parser.add_argument(
        "--output",
        type=str,
        default="data/expert_dataset.npz",
        help="Output path",
    )
    parser.add_argument(
        "--use_intentions",
        action="store_true",
        help="Inclure les intentions dans le dataset .npz",
    )
    args = parser.parse_args()
    generate_games(args.games, args.output, args.use_intentions)


if __name__ == "__main__":
    main()
