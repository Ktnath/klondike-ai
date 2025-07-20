from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrainingConfig:
    """Hyperparameters controlling the AlphaZero style training."""

    num_iterations: int = 1000
    num_episodes: int = 100
    temp_threshold: int = 15
    update_threshold: float = 0.6
    max_moves: int = 50
    num_mcts_sims: int = 25
    arena_compare: int = 40
    cpuct: float = 1.0
    checkpoint_interval: int = 20
    load_model: bool = False
    train_examples_history: int = 20


class _SimpleNet(nn.Module):
    def __init__(self, input_dim: int, action_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc_pi = nn.Linear(128, action_size)
        self.fc_v = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa: D401
        x = F.relu(self.fc1(x))
        pi = torch.softmax(self.fc_pi(x), dim=-1)
        v = torch.tanh(self.fc_v(x))
        return pi, v


class NeuralNet:
    """Minimal neural network wrapper used for training."""

    def __init__(self, input_shape: int, action_size: int, lr: float = 1e-3) -> None:
        self.model = _SimpleNet(input_shape, action_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, examples: List[Tuple[List[float], List[float], float]]) -> None:
        """Train the network on provided examples."""
        self.model.train()
        for board, pi, v in examples:
            board_t = torch.tensor(board, dtype=torch.float32)
            pi_t = torch.tensor(pi, dtype=torch.float32)
            v_t = torch.tensor([v], dtype=torch.float32)
            self.optimizer.zero_grad()
            pi_pred, v_pred = self.model(board_t)
            loss_pi = F.mse_loss(pi_pred, pi_t)
            loss_v = F.mse_loss(v_pred.squeeze(), v_t)
            loss = loss_pi + loss_v
            loss.backward()
            self.optimizer.step()

    def predict(self, board: List[float]) -> Tuple[List[float], float]:  # noqa: D401
        self.model.eval()
        board_t = torch.tensor(board, dtype=torch.float32)
        with torch.no_grad():
            pi, v = self.model(board_t)
        return pi.tolist(), float(v.item())

    def save_checkpoint(self, folder: str, filename: str) -> None:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, folder: str, filename: str) -> None:
        path = os.path.join(folder, filename)
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


class Coach:
    """Coordinates self-play and neural network training."""

    def __init__(self, neural_net: NeuralNet, config: TrainingConfig) -> None:
        self.neural_net = neural_net
        self.config = config

    def learn(self) -> None:  # noqa: D401
        raise NotImplementedError("Training loop not implemented yet.")
