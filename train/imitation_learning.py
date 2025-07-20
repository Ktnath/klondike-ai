"""Train an imitation model from logged episodes."""
from __future__ import annotations

import argparse
import csv
import glob
import os
import logging
from ast import literal_eval
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    """Simple multi-layer perceptron."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


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


def load_data(directory: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load all episode csv files from a directory."""
    all_obs: List[torch.Tensor] = []
    all_actions: List[int] = []
    for file in sorted(glob.glob(os.path.join(directory, "episode_*.csv"))):
        obs, acts = load_csv_file(file)
        all_obs.extend(obs)
        all_actions.extend(acts)
    if not all_obs:
        raise FileNotFoundError(f"No episode files found in {directory}")
    X = torch.stack(all_obs)
    y = torch.tensor(all_actions, dtype=torch.long)
    return X, y


def train(dataset: TensorDataset, epochs: int, model_path: str) -> None:
    """Train the imitation model and save it."""
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    input_dim = dataset.tensors[0].shape[1]
    num_actions = int(dataset.tensors[1].max().item()) + 1

    model = MLP(input_dim, num_actions)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(1)
            acc = (preds == y_batch).float().mean().item()
            logging.info("Epoch %d Batch Acc: %.3f", epoch, acc)

            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        logging.info("Epoch %d Average Loss: %.4f", epoch, avg_loss)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logging.info("Model saved to %s", model_path)


def evaluate(model_path: str, dataset: TensorDataset) -> None:
    """Evaluate a saved model on a dataset."""
    input_dim = dataset.tensors[0].shape[1]
    num_actions = int(dataset.tensors[1].max().item()) + 1
    model = MLP(input_dim, num_actions)
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
    parser = argparse.ArgumentParser(description="Imitation learning from episodes")
    parser.add_argument("--episodes_dir", type=str, default="logs/episodes", help="Directory with episode CSVs")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--model_path", type=str, default="models/imitation_model.pth", help="Where to save the model")
    parser.add_argument("--test", type=str, help="Optional test CSV file")
    args = parser.parse_args()

    X, y = load_data(args.episodes_dir)
    dataset = TensorDataset(X, y)
    train(dataset, args.epochs, args.model_path)

    if args.test:
        test_obs, test_actions = load_csv_file(args.test)
        test_data = TensorDataset(torch.stack(test_obs), torch.tensor(test_actions, dtype=torch.long))
        evaluate(args.model_path, test_data)

