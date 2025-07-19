"""Evaluate a trained DQN model on the Klondike environment."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch

from env.klondike_env import KlondikeEnv
from train_dqn import DQN


def evaluate(model_path: str, episodes: int) -> dict[str, Any]:
    """Run evaluation for a number of episodes using a greedy policy."""
    env = KlondikeEnv()
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(input_dim, action_dim)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    policy_net.load_state_dict(state_dict)
    policy_net.eval()

    total_reward = 0.0
    wins = 0

    for _ in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            valid_actions = env.get_valid_actions()
            with torch.no_grad():
                q_values = policy_net(torch.tensor(state, dtype=torch.float32))
                q_valid = q_values[valid_actions]
                action = valid_actions[int(torch.argmax(q_valid).item())]

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state

        total_reward += episode_reward
        if env.state.is_won():
            wins += 1

    avg_reward = total_reward / episodes
    results = {"episodes": episodes, "wins": wins, "avg_reward": avg_reward}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN model")
    parser.add_argument(
        "--model", type=str, default="models/dqn_final.pth", help="Path to model"
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--output", type=str, default="results/eval_results.json", help="Output JSON"
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = evaluate(args.model, args.episodes)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation on {args.episodes} episodes")
    win_rate = 100 * results["wins"] / results["episodes"] if results["episodes"] else 0.0
    print(f"Win rate: {win_rate:.2f} %")
    print(f"Average reward: {results['avg_reward']:.2f}")


if __name__ == "__main__":
    main()
