import argparse
import csv
import json
import os
from collections import Counter

import numpy as np
import torch
from tqdm import trange

# Automatically patched for modular project structure via bootstrap.py
from bootstrap import *

from env.klondike_env import KlondikeEnv
from train.train_dqn import DQN, DuelingDQN
from self_play_generate import _infer_intention
from intention_utils import simplify_intention
from klondike_core import move_from_index, is_won


def load_model(path: str, input_dim: int, action_dim: int) -> torch.nn.Module:
    """Load DQN model from ``path`` trying dueling then basic architecture."""
    model = DuelingDQN(input_dim, action_dim)
    try:
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
    except Exception:
        model = DQN(input_dim, action_dim)
        model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def evaluate(model_path: str, episodes: int, use_intentions: bool) -> dict:
    env = KlondikeEnv()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = load_model(model_path, obs_dim, action_dim)

    rewards: list[float] = []
    moves_per_game: list[int] = []
    wins = 0
    invalid = 0
    intentions = Counter()

    for _ in trange(episodes, desc="episodes"):
        state = env.reset()
        done = False
        ep_reward = 0.0
        move_count = 0
        current_state = env.state
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            with torch.no_grad():
                q = policy(torch.tensor(state, dtype=torch.float32))
                q_valid = q[valid]
                action = valid[int(torch.argmax(q_valid).item())]
            mv_str = move_from_index(action)
            next_state, reward, done, info = env.step(action)
            if use_intentions and mv_str is not None:
                intent = simplify_intention(
                    _infer_intention(current_state, mv_str, env.state)
                )
                intentions[intent] += 1
            if not info.get("valid", True):
                invalid += 1
            ep_reward += float(reward)
            move_count += 1
            state = next_state
            current_state = env.state
        rewards.append(ep_reward)
        moves_per_game.append(move_count)
        encoded = json.loads(env.state)["encoded"]
        if bool(is_won(encoded)):
            wins += 1

    total_moves = sum(moves_per_game)
    result = {
        "episodes": episodes,
        "wins": wins,
        "win_rate": 100 * wins / episodes if episodes else 0.0,
        "avg_moves": total_moves / episodes if episodes else 0.0,
        "invalid_rate": invalid / total_moves if total_moves else 0.0,
        "avg_reward": sum(rewards) / episodes if episodes else 0.0,
        "intentions": dict(intentions),
        "rewards": rewards,
    }
    return result


def export_results(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.lower().endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    elif path.lower().endswith(".csv"):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for key, val in data.items():
                if key != "rewards":
                    writer.writerow([key, val])
    else:
        raise ValueError("Unsupported output format")


def plot_metrics(results: dict, output: str | None) -> None:
    import matplotlib.pyplot as plt

    rewards = results.get("rewards", [])
    if rewards:
        plt.figure()
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward per Episode")
        plt.tight_layout()
        if output:
            plt.savefig(os.path.splitext(output)[0] + "_rewards.png")
        else:
            plt.show()

    intents = results.get("intentions", {})
    if intents:
        labels, counts = zip(*sorted(intents.items(), key=lambda x: -x[1]))
        plt.figure()
        plt.bar(range(len(labels)), counts)
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.title("Intentions distribution")
        plt.tight_layout()
        if output:
            plt.savefig(os.path.splitext(output)[0] + "_intentions.png")
        else:
            plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model_path", required=True, help="Path to model file")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of games")
    parser.add_argument(
        "--use_intentions", action="store_true", help="Track intentions"
    )
    parser.add_argument("--output", type=str, help="Output JSON or CSV path")
    args = parser.parse_args()

    results = evaluate(args.model_path, args.episodes, args.use_intentions)
    if args.output:
        export_results(results, args.output)
    print(json.dumps({k: v for k, v in results.items() if k != "rewards"}, indent=2))
    plot_metrics(results, args.output)


if __name__ == "__main__":
    main()
