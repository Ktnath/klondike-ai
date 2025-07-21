"""Train a simple DQN agent on the Klondike environment."""


from __future__ import annotations

import argparse
import os
import random
import csv
import logging
from typing import Tuple, List
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json

try:
    from klondike_core import solve_klondike, move_index
except Exception:  # pragma: no cover
    solve_klondike = None
    move_index = None

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.klondike_env import KlondikeEnv
from env.reward import is_critical_move
from utils.config import load_config
from dagger_dataset import DaggerDataset


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


class DuelingDQN(nn.Module):
    """Dueling architecture for approximating Q-values."""

    def __init__(self, input_dim: int, action_dim: int) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q


class ReplayBuffer:
    """Prioritized experience replay buffer."""

    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool, float]] = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(
        self, *transition: Tuple[np.ndarray, int, float, np.ndarray, bool, float]
    ) -> None:
        """Store a transition with maximal priority."""
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample a batch of transitions with importance-sampling weights."""
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]
        probs = prios**self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones, prio_w = zip(*samples)
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return (
            torch.from_numpy(np.array(states)).float(),
            torch.from_numpy(np.array(actions)).long(),
            torch.from_numpy(np.array(rewards)).float(),
            torch.from_numpy(np.array(next_states)).float(),
            torch.from_numpy(np.array(dones)).float(),
            torch.from_numpy(np.array(weights)).float(),
            torch.from_numpy(np.array(prio_w)).float(),
            indices,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: torch.Tensor) -> None:
        """Update priorities of sampled transitions."""
        for idx, err in zip(indices, td_errors.detach().cpu().numpy()):
            self.priorities[idx] = float(abs(err)) + 1e-6

    def __len__(self) -> int:
        return len(self.buffer)


def train(config) -> None:
    """Train a DQN agent using parameters from the config."""
    episodes = config.training.episodes
    env = KlondikeEnv()
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_type = getattr(config.model, "type", "dqn")
    model_cls = DuelingDQN if model_type == "dueling" else DQN

    policy_net = model_cls(input_dim, action_dim).to(device)
    target_net = model_cls(input_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=config.training.learning_rate)

    # Configuration options that may be absent from config.yaml
    expert_dataset_path = getattr(config, "expert_dataset", "data/expert_dataset.jsonl")
    dagger_dataset_path = getattr(config, "dagger_dataset", "data/dagger_buffer.jsonl")
    use_imitation = bool(getattr(config, "imitation_learning", False))
    use_dagger = bool(getattr(config, "dagger", False))
    use_weighting = bool(getattr(config, "critical_weighting", False))

    if use_imitation and os.path.exists(expert_dataset_path):
        logging.info("Pretraining from expert dataset %s", expert_dataset_path)
        obs_list: List[List[float]] = []
        act_list: List[int] = []
        w_list: List[float] = []
        with open(expert_dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                obs_list.append(d["observation"])
                act_list.append(d["action"])
                w_list.append(2.0 if d.get("is_critical") else 1.0)
        if obs_list:
            dataset = torch.utils.data.TensorDataset(
                torch.tensor(obs_list, dtype=torch.float32),
                torch.tensor(act_list, dtype=torch.long),
                torch.tensor(w_list, dtype=torch.float32),
            )
            loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
            for _ in range(3):
                for bx, by, bw in loader:
                    bx, by, bw = bx.to(device), by.to(device), bw.to(device)
                    optimizer.zero_grad()
                    logits = policy_net(bx)
                    loss = F.cross_entropy(logits, by, reduction="none")
                    (loss * bw).mean().backward()
                    optimizer.step()
            target_net.load_state_dict(policy_net.state_dict())

    per_cfg = config.training.get("per", {}) if hasattr(config.training, "get") else {}
    per_alpha = per_cfg.get("alpha", 0.6)
    per_beta = per_cfg.get("beta", 0.4)

    buffer = ReplayBuffer(config.training.buffer_size, alpha=per_alpha)

    dagger_ds = DaggerDataset(dagger_dataset_path)
    batch_size = config.training.batch_size
    gamma = config.training.gamma
    epsilon = config.training.epsilon.start
    epsilon_min = config.training.epsilon.min
    epsilon_decay = config.training.epsilon.decay
    target_update = config.training.target_update_freq
    log_interval = config.logging.log_interval
    save_interval = config.logging.save_interval
    log_path = getattr(config.logging, "log_path", "results/train_log.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    csv_logger = csv.writer(log_file)
    csv_logger.writerow(["episode", "reward", "loss", "epsilon", "win_rate"])

    ep_logging = getattr(config.logging, "enable_logging", False)
    episode_dir = os.path.join("logs", "episodes")
    if ep_logging:
        os.makedirs(episode_dir, exist_ok=True)
        existing = [
            int(f.split("_")[-1].split(".")[0])
            for f in os.listdir(episode_dir)
            if f.startswith("episode_") and f.endswith(".csv")
        ]
        next_ep = max(existing) if existing else 0
    save_path = config.model.save_path
    global_step = 0
    wins = 0

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    loss = torch.tensor(0.0)
    for episode in range(1, episodes + 1):
        logging.info("Starting episode %d", episode)
        seed = str(random.randint(0, 2**32 - 1))
        state = env.reset(seed)
        expert_moves = []
        if use_dagger and solve_klondike and move_index:
            try:
                sol = json.loads(solve_klondike(seed))
                expert_moves = [move_index(m) for m in sol.get("moves", [])]
            except Exception as exc:
                logging.warning("Solver failed: %s", exc)
        expert_step = 0
        done = False
        episode_reward = 0.0
        if ep_logging:
            next_ep += 1
            ep_path = os.path.join(episode_dir, f"episode_{next_ep}.csv")
            while os.path.exists(ep_path):
                next_ep += 1
                ep_path = os.path.join(episode_dir, f"episode_{next_ep}.csv")
            ep_file = open(ep_path, "w", newline="", encoding="utf-8")
            ep_file.write(f"# model: {os.path.basename(save_path)}\n")
            ep_writer = csv.DictWriter(
                ep_file,
                fieldnames=[
                    "step",
                    "observation",
                    "action",
                    "reward",
                    "done",
                    "epsilon",
                    "cumulative_reward",
                ],
            )
            ep_writer.writeheader()
            step_num = 0

        while not done:
            current_state = state
            valid_actions = env.get_valid_actions()
            if random.random() < epsilon:
                action = random.choice(valid_actions)
                logging.debug(
                    "Episode %d step %d: random action %s (epsilon %.3f)",
                    episode,
                    step_num if ep_logging else global_step,
                    action,
                    epsilon,
                )
            else:
                with torch.no_grad():
                    q_values = policy_net(
                        torch.tensor(state, dtype=torch.float32, device=device)
                    )
                    q_valid = q_values[valid_actions]
                    action = valid_actions[int(torch.argmax(q_valid).item())]
                logging.debug(
                    "Episode %d step %d: policy action %s",
                    episode,
                    step_num if ep_logging else global_step,
                    action,
                )

            next_state, reward, done, info = env.step(action)
            logging.debug(
                "Episode %d step %d: reward %.2f done %s",
                episode,
                step_num if ep_logging else global_step,
                reward,
                done,
            )
            episode_reward += reward
            if use_dagger and expert_step < len(expert_moves):
                expert_action = expert_moves[expert_step]
                if action != expert_action:
                    dagger_ds.add(current_state.tolist(), expert_action)
                expert_step += 1
            if ep_logging:
                ep_writer.writerow(
                    {
                        "step": step_num,
                        "observation": current_state.tolist(),
                        "action": action,
                        "reward": reward,
                        "done": done,
                        "epsilon": epsilon,
                        "cumulative_reward": episode_reward,
                    }
                )
                ep_file.flush()
                step_num += 1
            priority = 2.0 if use_weighting and is_critical_move(state, next_state) else 1.0
            buffer.push(state, action, reward, next_state, done, priority)
            state = next_state

            if len(buffer) >= batch_size:
                sample = buffer.sample(batch_size, beta=per_beta)
                (
                    b_states,
                    b_actions,
                    b_rewards,
                    b_next_states,
                    b_dones,
                    weights,
                    prio_w,
                    idxs,
                ) = sample
                b_states = b_states.to(device)
                b_actions = b_actions.to(device)
                b_rewards = b_rewards.to(device)
                b_next_states = b_next_states.to(device)
                b_dones = b_dones.to(device)
                weights = weights.to(device)
                prio_w = prio_w.to(device)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        q_values = (
                            policy_net(b_states)
                            .gather(1, b_actions.unsqueeze(1))
                            .squeeze(1)
                        )
                        with torch.no_grad():
                            next_actions = policy_net(b_next_states).argmax(
                                dim=1, keepdim=True
                            )
                            target_q = (
                                target_net(b_next_states)
                                .gather(1, next_actions)
                                .squeeze(1)
                            )
                            expected_q = b_rewards + gamma * (1 - b_dones) * target_q
                        td_errors = expected_q - q_values
                        loss = F.smooth_l1_loss(q_values, expected_q, reduction="none")
                        loss = (loss * weights * prio_w).mean()
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(
                        policy_net.parameters(), max_norm=1.0
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    q_values = (
                        policy_net(b_states)
                        .gather(1, b_actions.unsqueeze(1))
                        .squeeze(1)
                    )
                    with torch.no_grad():
                        next_actions = policy_net(b_next_states).argmax(
                            dim=1, keepdim=True
                        )
                        target_q = (
                            target_net(b_next_states).gather(1, next_actions).squeeze(1)
                        )
                        expected_q = b_rewards + gamma * (1 - b_dones) * target_q
                    td_errors = expected_q - q_values
                    loss = F.smooth_l1_loss(q_values, expected_q, reduction="none")
                    loss = (loss * weights * prio_w).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        policy_net.parameters(), max_norm=1.0
                    )
                    optimizer.step()

                logging.debug("Updated weights with loss %.4f", loss.item())

                buffer.update_priorities(idxs, td_errors)

                if global_step % target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            global_step += 1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        logging.debug("Epsilon decayed to %.3f", epsilon)
        if env.state.is_won():
            wins += 1

        if episode % log_interval == 0:
            win_rate = 100.0 * wins / episode
            loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
            logging.info(
                "Episode %d | Reward: %.2f | Loss: %.4f | Epsilon: %.3f | WinRate: %.2f %%",
                episode,
                episode_reward,
                loss_val,
                epsilon,
                win_rate,
            )
            csv_logger.writerow([episode, episode_reward, loss_val, epsilon, win_rate])

        if episode % save_interval == 0:
            base, ext = os.path.splitext(save_path)
            path = f"{base}_{episode}{ext}"
            torch.save(policy_net.state_dict(), path)
            logging.info("Saved checkpoint to %s", path)

        if ep_logging:
            ep_file.close()

        logging.info(
            "Finished episode %d with total reward %.2f", episode, episode_reward
        )

    torch.save(policy_net.state_dict(), save_path)
    logging.info("Model saved to %s", save_path)
    log_file.close()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
    )

    config = load_config()
    parser = argparse.ArgumentParser(description="Train DQN on Klondike")
    parser.add_argument(
        "--episodes",
        type=int,
        default=config.training.episodes,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--log",
        "--log-level",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level",
    )
    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    config.training.episodes = args.episodes
    train(config)
