import argparse
import json
import os
import random
from typing import Dict, List
from tqdm import trange
import numpy as np

from env.klondike_env import KlondikeEnv
from env.reward import compute_reward
from env.state_utils import (
    get_hidden_cards,
    count_empty_columns,
    extract_foundations,
)



try:
    from klondike_core import solve_klondike, move_index, shuffle_seed
except Exception:  # pragma: no cover - fallback when extension missing
    from klondike_core import solve_klondike, move_index
    shuffle_seed = None


def generate_games(num_games: int, output: str) -> None:
    """Generate expert dataset using the optimal solver."""
    os.makedirs(os.path.dirname(output), exist_ok=True)
    transitions: List[Dict] = []
    for _ in trange(num_games, desc="games"):
        seed = str(shuffle_seed()) if shuffle_seed else str(random.randint(0, 2**32 - 1))
        solution = json.loads(solve_klondike(seed))
        moves: List[str] = solution.get("moves", [])
        env = KlondikeEnv(seed=seed)
        env.reset(seed)
        prev_state = env.state
        for mv_json in moves:
            action = move_index(mv_json)
            _, rew, done, _ = env.step(action)

            prev_state_dict = json.loads(prev_state)
            next_state_dict = json.loads(env.state)

            tags = {
                "reveals_hidden_card": len(get_hidden_cards(next_state_dict)) < len(get_hidden_cards(prev_state_dict)),
                "frees_column": count_empty_columns(next_state_dict) > count_empty_columns(prev_state_dict),
                "foundation_progress": sum(extract_foundations(next_state_dict).values()) > sum(extract_foundations(prev_state_dict).values()),
                "is_critical": compute_reward(prev_state, action, env.state, done) > 0,
            }

            transitions.append(
                {
                    "state": prev_state_dict,
                    "action": {
                        "index": int(action),
                        "semantic_tags": tags,
                    },
                    "next_state": next_state_dict,
                    "reward": float(rew),
                    "done": bool(done),
                }
            )

            prev_state = env.state
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
