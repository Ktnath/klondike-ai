"""Utility for storing DAgger generated samples."""
from __future__ import annotations

import json
import os
from typing import List, Tuple

import torch


class DaggerDataset:
    """Simple JSONL based storage for DAgger transitions."""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def add(self, observation: List[float], action: int) -> None:
        """Append one sample to the dataset."""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"observation": observation, "action": action}) + "\n")

    def load(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load all samples as tensors."""
        if not os.path.exists(self.path):
            return torch.empty(0), torch.empty(0)
        obs: List[List[float]] = []
        acts: List[int] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                obs.append(d["observation"])
                acts.append(d["action"])
        return torch.tensor(obs, dtype=torch.float32), torch.tensor(acts, dtype=torch.long)
