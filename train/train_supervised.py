from __future__ import annotations

import argparse
import logging
import os
import sys

import torch

# Allow running as standalone script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train.train_dqn import DQN, load_dataset  # type: ignore
from train.intention_embedding import IntentionEncoder
from utils.training import log_epoch_metrics
from utils.config import load_config


DEFAULT_MODEL_PATH = "model.pt"


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Supervised training with intentions")
    parser.add_argument("--dataset", type=str, required=True, help="Path to .npz dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--output", type=str, default=DEFAULT_MODEL_PATH, help="Output model file")
    args = parser.parse_args()

    config = load_config()

    obs_arr, actions_arr, intents_arr = load_dataset(args.dataset)
    assert intents_arr is not None, "Dataset must contain 'intentions'"
    assert obs_arr.ndim == 2, "Observations should be a 2D array"
    assert obs_arr.shape[1] == 160, f"Expected observation dimension 160, got {obs_arr.shape[1]}"
    assert len(obs_arr) == len(actions_arr) == len(intents_arr)

    emb_dim = None
    if config.get("intention_embedding", {}).get("type") == "embedding":
        emb_dim = int(config.intention_embedding.get("dimension", 4))
    encoder = IntentionEncoder(embedding_dim=emb_dim)
    encoder.fit([str(i) for i in intents_arr])
    intent_vecs = encoder.encode_batch([str(i) for i in intents_arr])
    combine_mode = config.intention_embedding.get("combine_mode", "concat")

    obs_tensor = torch.tensor(obs_arr, dtype=torch.float32)
    if combine_mode == "concat":
        X = torch.cat([obs_tensor, intent_vecs], dim=1)
    else:
        if obs_tensor.shape[1] != intent_vecs.shape[1]:
            raise ValueError("Observation and intention dims must match for add/multiply")
        if combine_mode == "add":
            X = obs_tensor + intent_vecs
        elif combine_mode == "multiply":
            X = obs_tensor * intent_vecs
        else:
            raise ValueError(f"Unknown combine mode: {combine_mode}")

    assert X.shape[1] == 160, f"Input dimension after concatenation should be 160, got {X.shape[1]}"
    y = torch.tensor(actions_arr, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model = DQN(input_dim=160, action_dim=96)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
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

    os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None
    torch.save(model.state_dict(), args.output)
    logging.info("Model saved to %s", args.output)


if __name__ == "__main__":
    main()
