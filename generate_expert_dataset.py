import argparse
import json
import os
import random
from typing import List
from tqdm import trange

from env.klondike_env import KlondikeEnv
from env.reward import is_critical_move

try:
    from klondike_core import solve_klondike, move_index, shuffle_seed
except Exception:  # pragma: no cover - fallback when extension missing
    from klondike_core import solve_klondike, move_index
    shuffle_seed = None


def generate_games(num_games: int, output: str) -> None:
    """Generate expert dataset using the optimal solver."""
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for _ in trange(num_games, desc="games"):
            seed = str(shuffle_seed()) if shuffle_seed else str(random.randint(0, 2**32 - 1))
            solution = json.loads(solve_klondike(seed))
            moves: List[str] = solution.get("moves", [])
            env = KlondikeEnv(seed=seed)
            obs = env.reset(seed)
            prev_state = env.state
            for mv_json in moves:
                action = move_index(mv_json)
                next_obs, _, done, _ = env.step(action)
                is_crit = is_critical_move(prev_state, env.state)
                record = {
                    "observation": obs.tolist(),
                    "action": action,
                    "is_critical": is_crit,
                }
                f.write(json.dumps(record) + "\n")
                obs = next_obs
                prev_state = env.state
                if done:
                    break


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate expert dataset")
    parser.add_argument("--games", type=int, default=100, help="Number of games")
    parser.add_argument(
        "--output", type=str, default="data/expert_dataset.jsonl", help="Output path"
    )
    args = parser.parse_args()
    generate_games(args.games, args.output)


if __name__ == "__main__":
    main()
