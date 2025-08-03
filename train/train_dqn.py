"""Train a simple DQN agent on the Klondike environment."""


from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import pprint
from ast import literal_eval
from glob import glob
from typing import List, Tuple
from collections import Counter

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

try:  # Optional dependency
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


try:
    from klondike_core import solve_klondike, move_index, is_won
except Exception:  # pragma: no cover
    solve_klondike = None
    move_index = None
    is_won = None


# Automatically patched for modular project structure via bootstrap.py
from bootstrap import *

from env.klondike_env import KlondikeEnv
from env.is_critical_move import is_critical_move
from utils.config import load_config, get_config_value, DotDict, get_input_dim

# Path used when saving the final trained model
DEFAULT_MODEL_PATH = "model.pt"

from utils.training import log_epoch_metrics, log_episode
from dagger_dataset import DaggerDataset
from train.plot_results import plot_metrics
from train.intention_embedding import IntentionEncoder
from core.validate_dataset import validate_npz_dataset


def _init_writer(config):
    """Initialize a TensorBoard SummaryWriter if enabled."""
    if SummaryWriter is None:
        return None
    use_tb = bool(get_config_value(config, "logging.tensorboard", False))
    if not use_tb:
        return None
    log_dir = os.path.join("runs", f"exp_{int(time.time())}")
    print(f"📝 TensorBoard actif → logs dans {log_dir}")
    return SummaryWriter(log_dir=log_dir)


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

    def __init__(self, capacity: int, alpha: float = 0.6, obs_dim: int = 160) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.obs_dim = obs_dim
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool, float]] = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(
        self, *transition: Tuple[np.ndarray, int, float, np.ndarray, bool, float]
    ) -> None:
        """Store a transition with maximal priority."""
        state, _, _, next_state, _, _ = transition
        if state is not None and len(state) != self.obs_dim:
            raise ValueError(
                f"State dimension must be {self.obs_dim}, got {len(state)}"
            )
        if next_state is not None and len(next_state) != self.obs_dim:
            raise ValueError(
                f"Next state dimension must be {self.obs_dim}, got {len(next_state)}"
            )
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
        mask = probs > 0
        if mask.any():
            probs = probs[mask]
            indices_pool = np.arange(len(self.buffer))[mask]
        else:
            probs = np.ones_like(prios)
            indices_pool = np.arange(len(self.buffer))
        probs_sum = probs.sum()
        if probs_sum == 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs_sum
        indices = np.random.choice(indices_pool, batch_size, p=probs)
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


def _load_npz(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load data from a NPZ expert file."""
    data = np.load(path, allow_pickle=True)
    obs = data["observations"].astype(np.float32)
    actions = data["actions"].astype(np.int64)
    intents = data["intentions"] if "intentions" in data else None
    return obs, actions, intents


def _load_csv_dir(path: str) -> Tuple[np.ndarray, np.ndarray, None]:
    """Load observations/actions from a directory of CSV episode logs."""
    obs_list: List[List[float]] = []
    act_list: List[int] = []
    for file in sorted(glob(os.path.join(path, "*.csv"))):
        with open(file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(line for line in f if not line.startswith("#"))
            for row in reader:
                obs = literal_eval(row["observation"])
                act = int(row["action"])
                obs_list.append(obs)
                act_list.append(act)
    if not obs_list:
        raise FileNotFoundError(f"No CSV files found in {path}")
    X = np.array(obs_list, dtype=np.float32)
    y = np.array(act_list, dtype=np.int64)
    return X, y, None


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load dataset from NPZ file or CSV directory."""
    if os.path.isfile(path) and path.endswith(".npz"):
        return _load_npz(path)
    return _load_csv_dir(path)


def is_won_state(json_state: str) -> bool:
    """Return True if the JSON-encoded state indicates a win."""
    try:
        data = json.loads(json_state)
    except Exception:
        return False
    if isinstance(data, dict):
        if "is_won" in data:
            try:
                return bool(data["is_won"])
            except Exception:  # pragma: no cover - malformed state
                return False
        encoded = data.get("encoded")
        if encoded is not None and is_won is not None:
            try:
                return bool(is_won(encoded))
            except Exception:  # pragma: no cover - engine failure
                return False
    return False


def _load_state_dict_checked(model: nn.Module, state: dict) -> None:
    """Safely load a state dict ensuring tensor shapes match."""
    model_state = model.state_dict()
    mismatched: List[str] = []
    for key, value in state.items():
        if key in model_state and value.shape != model_state[key].shape:
            mismatched.append(
                f"{key}: checkpoint {tuple(value.shape)} vs model {tuple(model_state[key].shape)}"
            )
    if mismatched:
        msg = "; ".join(mismatched)
        raise ValueError(f"Incompatible pretrained weights ({msg})")
    model.load_state_dict(state, strict=False)


def train_supervised(
    dataset_path: str,
    use_intentions: bool,
    epochs: int,
    model_path: str,
    log_path: str,
) -> None:
    """Train the DQN model in a supervised manner from a dataset."""
    config = load_config()
    if dataset_path.endswith(".npz"):
        valid, message = validate_npz_dataset(
            dataset_path,
            use_intentions=getattr(config.env, "use_intentions", False),
        )
        if not valid:
            raise ValueError(f"[ERROR] Dataset invalide : {message}")
        print(f"[CHECK] {message}")
    obs_arr, actions_arr, intents_arr = load_dataset(dataset_path)

    encoder = None
    if use_intentions and intents_arr is not None:
        emb_dim = None
        if config.get("intention_embedding", {}).get("type") == "embedding":
            emb_dim = int(config.intention_embedding.get("dimension", 4))
        encoder = IntentionEncoder(embedding_dim=emb_dim)
        encoder.fit([str(i) for i in intents_arr])
        intent_vecs = encoder.encode_batch([str(i) for i in intents_arr])
        combine_mode = config.intention_embedding.get("combine_mode", "concat")
        if combine_mode == "concat":
            X = torch.tensor(obs_arr, dtype=torch.float32)
            X = torch.cat([X, intent_vecs], dim=1)
        else:
            obs = torch.tensor(obs_arr, dtype=torch.float32)
            if obs.shape[1] != intent_vecs.shape[1]:
                raise ValueError(
                    "Observation and intention dims must match for add/multiply"
                )
            if combine_mode == "add":
                X = obs + intent_vecs
            elif combine_mode == "multiply":
                X = obs * intent_vecs
            else:
                raise ValueError(f"Unknown combine mode: {combine_mode}")
    else:
        X = torch.tensor(obs_arr, dtype=torch.float32)
    y = torch.tensor(actions_arr, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    input_dim = X.shape[1]
    expected_dim = get_input_dim(config)
    if input_dim != expected_dim:
        raise ValueError(
            f"Input dimension should be {expected_dim}, got {input_dim}"
        )
    num_actions = int(y.max().item()) + 1
    model = DQN(input_dim, num_actions)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    logger = csv.writer(log_file)
    logger.writerow(["epoch", "loss", "accuracy"])

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        for bx, by in loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * bx.size(0)
            preds = logits.argmax(1)
            correct += (preds == by).sum().item()
            total += bx.size(0)

        avg_loss = total_loss / total
        log_epoch_metrics(epoch, avg_loss, correct, total)
        acc = correct / total if total else 0.0
        logger.writerow([epoch, avg_loss, acc])

    log_file.close()
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logging.info("Model saved to %s", model_path)

    if use_intentions and intents_arr is not None:
        counts = Counter([str(i) for i in intents_arr])
        dominant = counts.most_common(1)[0][0]
        info_path = os.path.join(os.path.dirname(model_path), "intentions_info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump({"dominant_intention": dominant, "counts": counts}, f, ensure_ascii=False, indent=2)
        logging.info("Intentions info saved to %s", info_path)

    # Plot training curves
    try:
        import pandas as pd

        df = pd.read_csv(log_path)
        plot_path = os.path.join(os.path.dirname(model_path), "training_plot.png")
        plot_metrics(df, plot_path)
        logging.info("Plot saved to %s", plot_path)
    except Exception as exc:  # pragma: no cover - plotting failures shouldn't stop training
        logging.warning("Could not generate plot: %s", exc)


def train(config, *, force_dim_check: bool = False) -> None:
    """Train a DQN agent using parameters from the config."""
    episodes = get_config_value(config, "training.episodes", 10000)
    use_int = bool(get_config_value(config, "env.use_intentions", True))
    env = KlondikeEnv(use_intentions=use_int)
    input_dim = env.observation_space.shape[0]
    expected_dim = get_input_dim(config)
    if input_dim != expected_dim:
        raise ValueError(
            f"Environment should provide {expected_dim}-dim observations, got {input_dim}"
        )
    if force_dim_check:
        logging.info("Observation dim: %d", input_dim)
    action_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = logging.getLogger(__name__)
    max_episode_steps = 1000

    model_type = get_config_value(config, "model.type", "dqn")
    model_cls = DuelingDQN if model_type == "dueling" else DQN

    policy_net = model_cls(input_dim, action_dim).to(device)
    target_net = model_cls(input_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    pretrained_path = getattr(getattr(config, "model", DotDict({})), "pretrained_path", None)
    if pretrained_path and os.path.exists(pretrained_path):
        state = torch.load(pretrained_path, map_location=device)
        first_key = next(iter(policy_net.state_dict()))
        if first_key in state and state[first_key].shape != policy_net.state_dict()[first_key].shape:
            logging.error(
                "Pretrained weights dimension %s incompatible with current configuration (%s)",
                state[first_key].shape,
                policy_net.state_dict()[first_key].shape,
            )
            raise ValueError(
                f"Pretrained model input_dim {state[first_key].shape[1]} does not match expected {policy_net.state_dict()[first_key].shape[1]}"
            )
        if force_dim_check:
            logging.info("Loaded state dict keys: %d", len(state))
        _load_state_dict_checked(policy_net, state)
        target_net.load_state_dict(policy_net.state_dict())
        print("✅ Chargement du modèle pré-entraîné")
    else:
        if pretrained_path:
            print(f"❌ Modèle introuvable : {pretrained_path}")

    lr = get_config_value(config, "training.learning_rate", 0.00025)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    # Configuration options that may be absent from config.yaml
    expert_dataset_path = get_config_value(config, "expert_dataset", "data/expert_dataset.npz")
    dagger_dataset_path = get_config_value(config, "dagger_dataset", "data/dagger_buffer.jsonl")
    use_imitation = bool(get_config_value(config, "imitation_learning", False))
    use_dagger = bool(get_config_value(config, "dagger", False))
    use_weighting = bool(get_config_value(config, "critical_weighting", False))

    if use_imitation and os.path.exists(expert_dataset_path):
        logging.info("Pretraining from expert dataset %s", expert_dataset_path)
        if expert_dataset_path.endswith(".jsonl"):
            obs_list: List[List[float]] = []
            act_list: List[int] = []
            w_list: List[float] = []
            with open(expert_dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    obs_list.append(d["observation"])
                    act_list.append(d["action"])
                    w_list.append(2.0 if d.get("is_critical") else 1.0)
            if not obs_list:
                loader = None
            else:
                dataset = torch.utils.data.TensorDataset(
                    torch.tensor(obs_list, dtype=torch.float32),
                    torch.tensor(act_list, dtype=torch.long),
                    torch.tensor(w_list, dtype=torch.float32),
                )
                loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        else:
            X_arr, y_arr, _ = load_dataset(expert_dataset_path)
            X = torch.tensor(X_arr, dtype=torch.float32)
            y = torch.tensor(y_arr, dtype=torch.long)
            w = torch.ones(len(y))
            dataset = torch.utils.data.TensorDataset(X, y, w)
            loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        if loader is not None:
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

    buffer_size = get_config_value(config, "training.buffer_size", 100000)
    buffer = ReplayBuffer(buffer_size, alpha=per_alpha, obs_dim=input_dim)

    dagger_ds = DaggerDataset(dagger_dataset_path)
    batch_size = get_config_value(config, "training.batch_size", 64)
    gamma = get_config_value(config, "training.gamma", 0.99)
    epsilon = get_config_value(config, "training.epsilon.start", 1.0)
    epsilon_min = get_config_value(config, "training.epsilon.min", 0.05)
    epsilon_decay = get_config_value(config, "training.epsilon.decay", 0.995)
    target_update = get_config_value(config, "training.target_update_freq", 1000)
    log_interval = get_config_value(config, "logging.log_interval", 100)
    save_interval = get_config_value(config, "logging.save_interval", 1000)
    log_path = get_config_value(config, "logging.log_path", "results/train_log.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    csv_logger = csv.writer(log_file)
    csv_logger.writerow([
        "episode",
        "reward",
        "loss",
        "epsilon",
        "win_rate",
        "avg_valid_moves",
    ])
    writer = _init_writer(config)

    ep_logging = get_config_value(config, "logging.enable_logging", False)
    episode_dir = os.path.join("logs", "episodes")
    if ep_logging:
        os.makedirs(episode_dir, exist_ok=True)
        existing = [
            int(f.split("_")[-1].split(".")[0])
            for f in os.listdir(episode_dir)
            if f.startswith("episode_") and f.endswith(".csv")
        ]
        next_ep = max(existing) if existing else 0
    global_step = 0
    wins = 0

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    save_dir = os.path.dirname(DEFAULT_MODEL_PATH)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    loss = torch.tensor(0.0)
    seed = str(random.randint(0, 2**32 - 1))
    state, _ = env.reset(seed=seed)  # migrated from gym to gymnasium
    logger.debug("Episode 1 reset with seed %s -> %s", seed, state)
    for episode in range(1, episodes + 1):
        logging.info("Starting episode %d", episode)
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
        steps_in_episode = 0
        valid_move_sum = 0
        episode_actions: List[int] = []
        expert_actions_taken: List[int] = []
        if writer and episode % 200 == 0:
            writer.add_embedding(
                torch.tensor(state, dtype=torch.float32).unsqueeze(0),
                metadata=[f"episode_{episode}"],
                global_step=episode,
            )
        if ep_logging:
            next_ep += 1
            ep_path = os.path.join(episode_dir, f"episode_{next_ep}.csv")
            while os.path.exists(ep_path):
                next_ep += 1
                ep_path = os.path.join(episode_dir, f"episode_{next_ep}.csv")
            ep_file = open(ep_path, "w", newline="", encoding="utf-8")
            ep_file.write(f"# model: {os.path.basename(DEFAULT_MODEL_PATH)}\n")
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

        while not done and steps_in_episode < max_episode_steps:
            current_state = state
            valid_actions = env.get_valid_actions()
            valid_move_sum += len(valid_actions)
            logger.debug(
                "Episode %d step %d: selecting action from %s",
                episode,
                steps_in_episode,
                valid_actions,
            )
            try:
                if random.random() < epsilon:
                    action = random.choice(valid_actions)
                    logger.debug(
                        "Episode %d step %d: random action %s (epsilon %.3f)",
                        episode,
                        steps_in_episode,
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
                    logger.debug(
                        "Episode %d step %d: policy action %s",
                        episode,
                        steps_in_episode,
                        action,
                    )
            except Exception as exc:
                logger.exception(
                    "Action selection failed at episode %d step %d: %s",
                    episode,
                    steps_in_episode,
                    exc,
                )
                break

            episode_actions.append(action)
            logger.debug(
                "Episode %d step %d: calling env.step(%s)",
                episode,
                steps_in_episode,
                action,
            )
            try:
                next_state, reward, terminated, truncated, info = env.step(action)  # migrated from gym to gymnasium
                done = terminated or truncated
            except Exception as exc:
                logger.exception(
                    "env.step failed at episode %d step %d: %s",
                    episode,
                    steps_in_episode,
                    exc,
                )
                break
            logger.debug(
                "Episode %d step %d: reward %.2f done %s",
                episode,
                steps_in_episode,
                reward,
                done,
            )

            if next_state is None or (hasattr(next_state, "__len__") and len(next_state) == 0):
                logger.error(
                    "Invalid observation at episode %d step %d: %s",
                    episode,
                    steps_in_episode,
                    next_state,
                )

            episode_reward += reward
            if use_dagger and expert_step < len(expert_moves):
                expert_action = expert_moves[expert_step]
                expert_actions_taken.append(expert_action)
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
            logger.debug(
                "Transition added: s=%s a=%s r=%.2f ns=%s d=%s",
                state,
                action,
                reward,
                next_state,
                done,
            )
            state = next_state

            steps_in_episode += 1
            if steps_in_episode >= max_episode_steps and not done:
                logger.warning(
                    "Episode %d exceeded max steps %d without done", 
                    episode, 
                    max_episode_steps,
                )
                done = True

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
            else:
                logger.debug(
                    "Buffer size %d < batch size %d, skipping training step",
                    len(buffer),
                    batch_size,
                )

        log_episode(episode, episode_reward, steps_in_episode)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        logging.debug("Epsilon decayed to %.3f", epsilon)
        if is_won_state(env.state):
            wins += 1
        win_rate = 100.0 * wins / episode
        if writer:
            writer.add_scalar("Reward/mean", episode_reward, episode)
            loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
            writer.add_scalar("Loss", loss_val, episode)
            writer.add_scalar("WinRate", win_rate, episode)
            action_counts = Counter(episode_actions)
            top5 = {f"move_{a}": c for a, c in action_counts.most_common(5)}
            if top5:
                writer.add_scalars("Actions/Frequency", top5, episode)
            if episode % 100 == 0:
                for name, param in policy_net.named_parameters():
                    writer.add_histogram(name, param, episode)
            if (
                use_dagger
                and expert_actions_taken
                and sns is not None
                and episode % 50 == 0
            ):
                cm = confusion_matrix(
                    expert_actions_taken[: len(episode_actions)],
                    episode_actions[: len(expert_actions_taken)],
                )
                fig = plt.figure()
                sns.heatmap(cm, annot=True)
                writer.add_figure("Confusion Matrix", fig, episode)
                plt.close(fig)

        if episode % log_interval == 0:
            # win_rate already computed above
            loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
            avg_valid = valid_move_sum / steps_in_episode if steps_in_episode else 0
            logging.info(
                "Episode %d | Reward: %.2f | Loss: %.4f | Epsilon: %.3f | WinRate: %.2f %% | AvgMoves: %.2f",
                episode,
                episode_reward,
                loss_val,
                epsilon,
                win_rate,
                avg_valid,
            )
            csv_logger.writerow([
                episode,
                episode_reward,
                loss_val,
                epsilon,
                win_rate,
                avg_valid,
            ])

        if episode % save_interval == 0:
            checkpoint_dir = os.path.join("models", "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            path = os.path.join(checkpoint_dir, f"checkpoint_{episode}.pt")
            torch.save(policy_net.state_dict(), path)
            logging.info("Saved checkpoint to %s", path)

        if ep_logging:
            ep_file.close()

        logging.info(
            "Finished episode %d with total reward %.2f", episode, episode_reward
        )
        seed = str(random.randint(0, 2**32 - 1))
        state, _ = env.reset(seed=seed)  # migrated from gym to gymnasium
        logger.debug(
            "Episode %d reset with seed %s -> %s", episode + 1, seed, state
        )

    torch.save(policy_net.state_dict(), DEFAULT_MODEL_PATH)
    logging.info(f"Mod\u00e8le final sauvegard\u00e9 sous {DEFAULT_MODEL_PATH}")
    log_file.close()
    if writer:
        writer.close()
    print("📊 Logs TensorBoard enrichis avec histogrammes et heatmaps")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
    )

    config = load_config()
    parser = argparse.ArgumentParser(description="Train DQN")
    parser.add_argument("--dataset", type=str, help="Path to NPZ or CSV dataset")
    parser.add_argument("--use_intentions", action="store_true", help="Use intention labels if available")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs for supervised training")
    parser.add_argument("--model_path", type=str, default="results/dqn_supervised.pth", help="Output model path")
    parser.add_argument("--log_path", type=str, default="results/train_log.csv", help="CSV log path")

    expert_default = get_config_value(config, "expert_dataset", "data/expert_dataset.npz")
    pretrained_default = get_config_value(config, "model.pretrained_path", DEFAULT_MODEL_PATH)

    parser.add_argument("--expert_dataset", type=str, default=None, help=f"Path to expert dataset (default: {expert_default})")
    parser.add_argument("--pretrained_model", type=str, default=None, help=f"Path to pretrained model (default: {pretrained_default})")
    parser.add_argument("--imitation_learning", action="store_true", help="Enable imitation learning")
    parser.add_argument("--dagger", action="store_true", help="Enable DAgger")

    parser.add_argument(
        "--episodes",
        type=int,
        default=get_config_value(config, "training.episodes", 10000),
        help="Number of RL episodes",
    )
    parser.add_argument(
        "--force-dim-check",
        action="store_true",
        help="Log input/output dimensions when loading models",
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

    if args.expert_dataset is not None:
        config.expert_dataset = args.expert_dataset
    else:
        config.expert_dataset = expert_default

    if args.pretrained_model is not None:
        if not hasattr(config, "model"):
            config.model = DotDict({})
        config.model.pretrained_path = args.pretrained_model
    else:
        if not hasattr(config, "model"):
            config.model = DotDict({})
        config.model.pretrained_path = pretrained_default
    if args.imitation_learning:
        config.imitation_learning = True
    if args.dagger:
        config.dagger = True

    logging.info("Using configuration:\n%s", pprint.pformat(config))

    if args.dataset:
        train_supervised(args.dataset, args.use_intentions, args.epochs, args.model_path, args.log_path)
    else:
        config.training.episodes = args.episodes
        train(config, force_dim_check=args.force_dim_check)
