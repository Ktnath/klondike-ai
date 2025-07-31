from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch

from utils.config import get_input_dim, load_config

# Allow running as standalone script from repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from ai.klondike_ai import NeuralNet
except Exception as exc:  # pragma: no cover - handle missing module
    raise ImportError(
        "Could not import NeuralNet from ai.klondike_ai. Make sure the file"
        " ai/klondike_ai.py exists and is on the PYTHONPATH"
    ) from exc

from utils.training import log_epoch_metrics


DEFAULT_MODEL_PATH = "model.pt"


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Supervised training with intentions")
    parser.add_argument("--dataset", type=str, required=True, help="Path to .npz dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--output", type=str, default=DEFAULT_MODEL_PATH, help="Output model file")
    parser.add_argument(
        "--use_intentions",
        action="store_true",
        help="Inclure les intentions lors de l’entraînement",
    )
    args = parser.parse_args()

    cfg = load_config()
    expected_dim = get_input_dim(cfg)

    data = np.load(args.dataset)
    X = data["observations"].astype(np.float32)
    y = data["actions"].astype(np.int64)

    if X.shape[1] != expected_dim:
        raise ValueError(f"Dataset dimension {X.shape[1]} does not match expected {expected_dim}")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    input_dim = X.shape[1]
    action_dim = int(y_tensor.max().item()) + 1
    model = NeuralNet(input_shape=input_dim, action_size=action_dim)
    optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-3)
    criterion = torch.nn.NLLLoss()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        for bx, by in loader:
            optimizer.zero_grad()
            probs, _ = model.model(bx)
            loss = criterion(torch.log(probs), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * bx.size(0)
            preds = probs.argmax(1)
            correct += (preds == by).sum().item()
            total += bx.size(0)
        avg_loss = total_loss / total
        log_epoch_metrics(epoch, avg_loss, correct, total)

    os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None
    torch.save(model.state_dict(), args.output)
    logging.info("Model saved to %s", args.output)


if __name__ == "__main__":
    main()
