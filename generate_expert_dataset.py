import argparse
import json
import os
import random
from typing import Dict, List

from intention_utils import simplify_intention, filter_ambiguous
from tqdm import trange
import numpy as np

# Automatically patched for modular project structure via bootstrap.py
from bootstrap import *

from klondike_core import (
    new_game,
    legal_moves,
    play_move,
    move_index,
    compute_base_reward_json,
    solve_klondike,
    shuffle_seed,
    encode_observation,
)


def generate_games(num_games: int, output: str) -> None:
    """Generate expert dataset using the optimal solver."""
    os.makedirs(os.path.dirname(output), exist_ok=True)

    observations: List[List[float]] = []
    actions: List[int] = []
    rewards: List[float] = []
    dones: List[bool] = []
    intentions: List[str] = []

    for _ in trange(num_games, desc="games"):
        seed = str(shuffle_seed()) if shuffle_seed else str(random.randint(0, 2**32 - 1))
        state = new_game(seed)
        solution = json.loads(solve_klondike(state))
        result = solution.get("result", [])
        # Some versions may return intentions separately
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

            obs = encode_observation(prev_state)
            action = move_index(mv_json)
            next_state, _ = play_move(prev_state, mv_json)
            reward = compute_base_reward_json(next_state)
            done = json.loads(next_state).get("is_won", False)

            observations.append(list(obs))
            actions.append(int(action))
            rewards.append(float(reward))
            dones.append(bool(done))
            intentions.append(simplify_intention(str(intention)))

            prev_state = next_state
            if done:
                break

    filtered = filter_ambiguous(intentions)
    mask = [i is not None for i in filtered]
    observations = [o for o, m in zip(observations, mask) if m]
    actions = [a for a, m in zip(actions, mask) if m]
    rewards = [r for r, m in zip(rewards, mask) if m]
    dones = [d for d, m in zip(dones, mask) if m]
    intentions = [i for i in filtered if i is not None]

    np.savez_compressed(
        output,
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.int64),
        rewards=np.array(rewards, dtype=np.float32),
        dones=np.array(dones, dtype=bool),
        intentions=np.array(intentions, dtype=object),
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
    args = parser.parse_args()
    generate_games(args.games, args.output)


if __name__ == "__main__":
    main()
