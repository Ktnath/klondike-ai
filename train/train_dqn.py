"""Train a simple DQN agent on the Klondike environment."""
from __future__ import annotations

import argparse
import os
import random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env.klondike_env import KlondikeEnv
from utils.config import load_config


class DQN(nn.Module):
    """Basic feed-forward network for approximating Q-values."""

    def __init__(self, input_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


class ReplayBuffer:
    """Simple replay buffer for experience replay."""

    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, *transition: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


def train(config) -> None:
    """Train a DQN agent using parameters from the config."""
    episodes = config.training.episodes
    env = KlondikeEnv()
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(input_dim, action_dim)
    target_net = DQN(input_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=config.training.learning_rate)
    criterion = nn.SmoothL1Loss()

    buffer = ReplayBuffer(config.training.buffer_size)
    batch_size = config.training.batch_size
    gamma = config.training.gamma
    epsilon = config.training.epsilon.start
    epsilon_min = config.training.epsilon.min
    epsilon_decay = config.training.epsilon.decay
    target_update = config.training.target_update_freq
    log_interval = config.logging.log_interval
    save_interval = config.logging.save_interval
    save_path = config.model.save_path
    global_step = 0
    wins = 0

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            valid_actions = env.get_valid_actions()
            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.tensor(state, dtype=torch.float32))
                    q_valid = q_values[valid_actions]
                    action = valid_actions[int(torch.argmax(q_valid).item())]

            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            buffer.push(state, action, reward, next_state, done)
            state = next_state

            if len(buffer) >= batch_size:
                (b_states, b_actions, b_rewards, b_next_states, b_dones) = buffer.sample(batch_size)
                q_values = policy_net(b_states).gather(1, b_actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    target_q = target_net(b_next_states).max(1)[0]
                    expected_q = b_rewards + gamma * (1 - b_dones) * target_q
                loss = criterion(q_values, expected_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if global_step % target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            global_step += 1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if env.state.is_won():
            wins += 1

        if episode % log_interval == 0:
            print(
                f"Episode {episode} | Reward: {episode_reward:.2f} | Loss: {loss.item() if 'loss' in locals() else 0:.4f} | "
                f"Epsilon: {epsilon:.3f} | Wins: {wins}"
            )

        if episode % save_interval == 0:
            base, ext = os.path.splitext(save_path)
            torch.save(policy_net.state_dict(), f"{base}_{episode}{ext}")

    torch.save(policy_net.state_dict(), save_path)


if __name__ == "__main__":
    config = load_config()
    parser = argparse.ArgumentParser(description="Train DQN on Klondike")
    parser.add_argument(
        "--episodes",
        type=int,
        default=config.training.episodes,
        help="Number of training episodes",
    )
    args = parser.parse_args()
    config.training.episodes = args.episodes
    train(config)
