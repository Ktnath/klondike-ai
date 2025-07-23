"""End-to-end test script for the Klondike environment.

This script plays a random game using the Gym environment.
"""
from __future__ import annotations

import random

from env.klondike_env import KlondikeEnv


MAX_STEPS = 50


def main() -> None:
    env = KlondikeEnv()
    env.reset()

    total_reward = 0.0
    done = False
    step_count = 0

    while not done and step_count < MAX_STEPS:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            print("No valid actions available.")
            break

        action = random.choice(valid_actions)
        _, reward, done, info = env.step(action)

        step_count += 1
        total_reward += reward
        print(f"Step {step_count}: reward={reward}, done={done}, info={info}")

    print("\nNombre de coups :", step_count)
    print("Score final :", total_reward)
    print("Partie gagnée :", "Oui" if done else "Non")


if __name__ == "__main__":
    main()
