"""Train an imitation model from logged episodes or expert NPZ dataset."""
from __future__ import annotations

import argparse
import csv
import glob
import os
import logging
from ast import literal_eval
from typing import List, Tuple, Optional

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from intention_utils import group_into_hierarchy
from utils.config import load_config, get_input_dim, get_config_value
from train.train_dqn import DQN

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
from collections import Counter
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
from utils.training import log_epoch_metrics
from env.klondike_env import KlondikeEnv
from core.validate_dataset import validate_npz_dataset

# Default expert dataset used when no --dataset argument is provided
DEFAULT_DATASET = os.path.join("data", "expert_dataset.npz")

# Automatically patched for modular project structure via bootstrap.py
from bootstrap import *


def _init_writer(config):
    """Initialize a TensorBoard writer if enabled in config."""
    if SummaryWriter is None:
        return None
    use_tb = bool(get_config_value(config, "logging.tensorboard", False))
    if not use_tb:
        return None
    log_dir = os.path.join("runs", f"exp_{int(time.time())}")
    print(f"📝 TensorBoard actif → logs dans {log_dir}")
    return SummaryWriter(log_dir=log_dir)


def _intention_to_index(value: str | int | None) -> int:
    """Map an intention label to the canonical index."""
    mapping = {
        "reveal": 0,
        "foundation": 1,
        "stack_move": 2,
        "king_to_empty": 3,
    }
    if isinstance(value, (int, np.integer)):
        idx = int(value)
        return idx if 0 <= idx < 4 else 0
    key = str(value).strip().lower().replace(" ", "_")
    for name, idx in mapping.items():
        if name in key:
            return idx
    return 0


def _intention_vector(value: str | int | None) -> torch.Tensor:
    vec = torch.zeros(4, dtype=torch.float32)
    vec[_intention_to_index(value)] = 1.0
    return vec


def load_csv_file(path: str) -> Tuple[List[torch.Tensor], List[int]]:
    """Load one episode csv and return observations and actions."""
    observations: List[torch.Tensor] = []
    actions: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        for row in reader:
            obs = literal_eval(row["observation"])
            observations.append(torch.tensor(obs, dtype=torch.float32))
            actions.append(int(row["action"]))
    return observations, actions


def load_npz_file(path: str, use_intentions: bool, use_hierarchy: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Load a NPZ dataset with optional intention labels."""
    data = np.load(path, allow_pickle=True)
    obs = torch.tensor(data["observations"], dtype=torch.float32)
    actions = torch.tensor(data["actions"], dtype=torch.long)
    intentions: Optional[torch.Tensor] = None

    if "intentions" in data:
        raw = data["intentions"]
        if raw.dtype.kind in {"U", "S", "O"}:
            values = raw.tolist()
            if use_hierarchy:
                values = group_into_hierarchy(values)
            idxs = torch.tensor([_intention_to_index(v) for v in values], dtype=torch.long)
        else:
            idxs = torch.tensor([_intention_to_index(int(v)) for v in raw], dtype=torch.long)

        if use_intentions:
            vecs = torch.nn.functional.one_hot(idxs, num_classes=4).float()
            obs = torch.cat([obs, vecs], dim=1)
        intentions = idxs

    return obs, actions, intentions


def load_data(
    source: str, use_intentions: bool = False, use_hierarchy: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Load dataset from CSV directory or NPZ file."""
    if os.path.isfile(source) and source.endswith(".npz"):
        return load_npz_file(source, use_intentions, use_hierarchy)

    all_obs: List[torch.Tensor] = []
    all_actions: List[int] = []
    for file in sorted(glob.glob(os.path.join(source, "episode_*.csv"))):
        obs, acts = load_csv_file(file)
        all_obs.extend(obs)
        all_actions.extend(acts)
    if not all_obs:
        raise FileNotFoundError(f"No episode files found in {source}")
    X = torch.stack(all_obs)
    y = torch.tensor(all_actions, dtype=torch.long)
    return X, y, None


def train(dataset: TensorDataset, epochs: int, model_path: str, intentions: Optional[torch.Tensor] = None) -> None:
    """Train the imitation model and save it."""
    cfg = load_config()
    expected_dim = get_input_dim(cfg)
    writer = _init_writer(cfg)

    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    if dataset.tensors[0].shape[1] != expected_dim:
        raise ValueError(
            f"Dataset dimension {dataset.tensors[0].shape[1]} does not match expected {expected_dim}"
        )
    num_actions = int(dataset.tensors[1].max().item()) + 1

    model = DQN(expected_dim, num_actions)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        correct = 0
        total = 0
        epoch_preds: List[int] = []
        epoch_labels: List[int] = []
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            epoch_preds.extend(preds.tolist())
            epoch_labels.extend(y_batch.tolist())

            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        log_epoch_metrics(epoch, avg_loss, correct, total)
        if intentions is not None:
            uniq, counts = torch.unique(intentions, return_counts=True)
            idx = counts.argmax().item()
            dominant = uniq[idx].item()
            logging.info("Epoch %d Dominant intention: %s (%d samples)", epoch, dominant, counts[idx].item())
        if writer:
            writer.add_scalar("Loss", avg_loss, epoch)
            acc = correct / total if total else 0
            writer.add_scalar("Accuracy", acc, epoch)
            action_counts = Counter(epoch_preds)
            top5 = {f"move_{a}": c for a, c in action_counts.most_common(5)}
            if top5 and epoch % 5 == 0:
                writer.add_scalars("Actions/Frequency", top5, epoch)
            if epoch % 50 == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param, epoch)
            if sns is not None and epoch % 10 == 0:
                cm = confusion_matrix(epoch_labels, epoch_preds)
                fig = plt.figure()
                sns.heatmap(cm, annot=True)
                writer.add_figure("Confusion Matrix", fig, epoch)
                plt.close(fig)
            if epoch % 100 == 0:
                writer.add_embedding(
                    dataset.tensors[0][0].unsqueeze(0),
                    metadata=[f"epoch_{epoch}"],
                    global_step=epoch,
                )

    dirpath = os.path.dirname(model_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logging.info("Model saved to %s", model_path)
    if writer:
        writer.close()
    print("📊 Logs TensorBoard enrichis avec histogrammes et heatmaps")


def fine_tune_model(
    model: DQN,
    dataset: TensorDataset,
    epochs: int,
    intentions: Optional[torch.Tensor] = None,
) -> None:
    """Fine tune a pre-trained model on a dataset with a scheduler."""
    cfg = load_config()
    expected_dim = get_input_dim(cfg)
    writer = _init_writer(cfg)

    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    if dataset.tensors[0].shape[1] != expected_dim:
        raise ValueError(
            f"Dataset dimension {dataset.tensors[0].shape[1]} does not match expected {expected_dim}"
        )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, epochs // 3), gamma=0.5
    )

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        correct = 0
        total = 0
        epoch_preds: List[int] = []
        epoch_labels: List[int] = []
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            epoch_loss += loss.item()
            epoch_preds.extend(preds.tolist())
            epoch_labels.extend(y_batch.tolist())
        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        log_epoch_metrics(epoch, avg_loss, correct, total)
        if writer:
            writer.add_scalar("Loss", avg_loss, epoch)
            acc = correct / total if total else 0
            writer.add_scalar("Accuracy", acc, epoch)
            action_counts = Counter(epoch_preds)
            top5 = {f"move_{a}": c for a, c in action_counts.most_common(5)}
            if top5 and epoch % 5 == 0:
                writer.add_scalars("Actions/Frequency", top5, epoch)
            if epoch % 50 == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param, epoch)
            if sns is not None and epoch % 10 == 0:
                cm = confusion_matrix(epoch_labels, epoch_preds)
                fig = plt.figure()
                sns.heatmap(cm, annot=True)
                writer.add_figure("Confusion Matrix", fig, epoch)
                plt.close(fig)
            if epoch % 100 == 0:
                writer.add_embedding(
                    dataset.tensors[0][0].unsqueeze(0),
                    metadata=[f"epoch_{epoch}"],
                    global_step=epoch,
                )
        if intentions is not None:
            uniq, counts = torch.unique(intentions, return_counts=True)
            idx = counts.argmax().item()
            dominant = uniq[idx].item()
            logging.info(
                "Epoch %d Dominant intention: %s (%d samples)",
                epoch,
                dominant,
                counts[idx].item(),
            )
    if writer:
        writer.close()
    print("📊 Logs TensorBoard enrichis avec histogrammes et heatmaps")


def reinforce_train(model: DQN, episodes: int, gamma: float = 0.99) -> None:
    """Simple REINFORCE fine-tuning on the Klondike environment."""
    cfg = load_config()
    use_int = bool(getattr(cfg.env, "use_intentions", False))
    env = KlondikeEnv(use_intentions=use_int)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for episode in range(1, episodes + 1):
        state, _ = env.reset()  # migrated from gym to gymnasium
        done = False
        states: List[torch.Tensor] = []
        actions: List[int] = []
        rewards: List[float] = []
        while not done:
            with torch.no_grad():
                logits = model(torch.tensor(state, dtype=torch.float32))
                probs = torch.softmax(logits, dim=0)
            action = int(torch.multinomial(probs, 1).item())
            next_state, reward, terminated, truncated, _ = env.step(action)  # migrated from gym to gymnasium
            done = terminated or truncated
            states.append(torch.tensor(state, dtype=torch.float32))
            actions.append(action)
            rewards.append(float(reward))
            state = next_state

        returns: List[float] = []
        R = 0.0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        logits = model(torch.stack(states))
        log_probs = torch.log_softmax(logits, dim=1)
        selected = log_probs[range(len(actions)), actions]
        loss = -(selected * returns_t).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logging.info("Episode %d RL Loss: %.4f", episode, loss.item())


def evaluate(model_path: str, dataset: TensorDataset) -> None:
    """Evaluate a saved model on a dataset."""
    cfg = load_config()
    expected_dim = get_input_dim(cfg)

    input_dim = dataset.tensors[0].shape[1]
    if input_dim != expected_dim:
        raise ValueError(f"Dataset dimension {input_dim} does not match expected {expected_dim}")
    num_actions = int(dataset.tensors[1].max().item()) + 1
    model = DQN(expected_dim, num_actions)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    loader = DataLoader(dataset, batch_size=64)
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            preds = logits.argmax(1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    acc = correct / total if total else 0.0
    logging.info("Test Accuracy: %.3f", acc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Imitation learning and fine-tuning")
    parser.add_argument("--episodes_dir", type=str, default="logs/episodes", help="Directory with episode CSVs")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--model_path", type=str, default="models/imitation_model.pth", help="Model to load")
    parser.add_argument("--output_path", type=str, default="model.pt", help="Output path")
    parser.add_argument("--dataset", type=str, help="Dataset for fine-tuning (.npz or CSV dir)")
    parser.add_argument("--test", type=str, help="Optional test dataset (.csv or .npz)")
    parser.add_argument("--use_intentions", action="store_true", help="Use intention labels if available")
    parser.add_argument(
        "--use_intention_hierarchy",
        action="store_true",
        help="Group intentions into higher-level categories",
    )
    parser.add_argument("--fine_tune", action="store_true", help="Enable dataset fine-tuning mode")
    parser.add_argument("--reinforce", action="store_true", help="Enable reinforcement fine-tuning")
    parser.add_argument("--hybrid", action="store_true", help="Combine imitation and RL")
    parser.add_argument("--episodes", type=int, default=50, help="RL episodes for fine-tuning")
    args = parser.parse_args()

    cfg = load_config()
    expected_dim = get_input_dim(cfg)
    use_int_cfg = bool(getattr(cfg.env, "use_intentions", False))

    dataset_path = args.dataset if args.dataset else DEFAULT_DATASET

    # Automatically enable fine tuning when using an NPZ dataset
    if dataset_path.endswith(".npz"):
        args.fine_tune = True

    if args.reinforce and not args.fine_tune:
        env = KlondikeEnv(use_intentions=use_int_cfg)
        input_dim = env.observation_space.shape[0]
        num_actions = env.action_space.n
        if input_dim != expected_dim:
            raise ValueError(f"Env dimension {input_dim} does not match expected {expected_dim}")
        model = DQN(expected_dim, num_actions)
        if os.path.exists(args.model_path):
            state = torch.load(args.model_path, map_location=torch.device("cpu"))
            model.load_state_dict(state)
        reinforce_train(model, args.episodes)
        dirpath = os.path.dirname(args.output_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        torch.save(model.state_dict(), args.output_path)
        logging.info("Model saved to %s", args.output_path)
    else:
        if dataset_path.endswith(".npz"):
            valid, message = validate_npz_dataset(
                dataset_path,
                use_intentions=getattr(cfg.env, "use_intentions", False),
            )
            if not valid:
                raise ValueError(f"[ERROR] Dataset invalide : {message}")
            print(f"[CHECK] {message}")
        X, y, intents = load_data(
            dataset_path, args.use_intentions, args.use_intention_hierarchy
        )
        if X.shape[1] != expected_dim:
            if os.path.isdir(dataset_path) and os.path.basename(dataset_path) == "episodes":
                logging.warning(
                    "Observations in %s have dimension %d instead of %d",
                    dataset_path,
                    X.shape[1],
                    expected_dim,
                )
            raise SystemExit(
                f"Dataset dimension {X.shape[1]} is invalid. Use --dataset {DEFAULT_DATASET}"
            )

        dataset = TensorDataset(X, y)
        if args.fine_tune:
            num_actions = int(y.max().item()) + 1
            model = DQN(expected_dim, num_actions)
            if os.path.exists(args.model_path):
                state = torch.load(args.model_path, map_location=torch.device("cpu"))
                model.load_state_dict(state)
            fine_tune_model(model, dataset, args.epochs, intents)
            if args.reinforce and args.hybrid:
                reinforce_train(model, args.episodes)
            dirpath = os.path.dirname(args.output_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            torch.save(model.state_dict(), args.output_path)
            logging.info("Model saved to %s", args.output_path)
        else:
            train(dataset, args.epochs, args.output_path, intents)

        if args.test:
            if args.test.endswith(".npz"):
                X_test, y_test, _ = load_data(
                    args.test, args.use_intentions, args.use_intention_hierarchy
                )
                test_dataset = TensorDataset(X_test, y_test)
            else:
                test_obs, test_actions = load_csv_file(args.test)
                test_dataset = TensorDataset(torch.stack(test_obs), torch.tensor(test_actions, dtype=torch.long))
            evaluate(args.output_path, test_dataset)

