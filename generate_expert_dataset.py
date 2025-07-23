import argparse
import json
import os
import random
from typing import Dict, List
from tqdm import trange
import numpy as np

from klondike_core import (
    new_game,
    legal_moves,
    play_move,
    move_index,
    compute_base_reward_json,
    solve_klondike,
    shuffle_seed,
)


def generate_games(num_games: int, output: str) -> None:
    """Generate expert dataset using the optimal solver."""
    os.makedirs(os.path.dirname(output), exist_ok=True)
    transitions: List[Dict] = []

    for _ in trange(num_games, desc="games"):
        seed = str(shuffle_seed()) if shuffle_seed else str(random.randint(0, 2**32 - 1))
        state = new_game(seed)
        solution = json.loads(solve_klondike(state))
        moves: List[str] = solution.get("result", [])
        prev_state = state
        for mv_json in moves:
            action = move_index(mv_json)
            next_state, _ = play_move(prev_state, mv_json)
            reward = compute_base_reward_json(next_state)

            done = json.loads(next_state).get("is_won", False)
            transitions.append(
                {
                    "state": json.loads(prev_state),
                    "action": int(action),
                    "next_state": json.loads(next_state),
                    "reward": float(reward),
                    "done": bool(done),
                }
            )

            prev_state = next_state
            if done:
                break

    np.savez_compressed(output, transitions=np.array(transitions, dtype=object))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate expert dataset")
    parser.add_argument("--games", type=int, default=100, help="Number of games")
    parser.add_argument(
        "--output",
        type=str,
        default="data/expert_dataset.npz",
        help="Output path",
    )
    args = parser.parse_args()
    generate_games(args.games, args.output)


if __name__ == "__main__":
    main()
