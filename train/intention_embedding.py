"""Intentions encoding utilities for ML models."""

from __future__ import annotations

import os
from typing import Dict, Sequence, Union

import numpy as np
import torch


class IntentionEncoder:
    """Encode intention strings as numeric vectors."""

    def __init__(self, embedding_dim: int | None = None, device: torch.device | None = None) -> None:
        """Create an encoder.

        Parameters
        ----------
        embedding_dim:
            Size of dense embedding. If ``None``, one-hot encoding is used.
        device:
            Optional device for returned tensors and the ``Embedding`` module.
        """
        self.embedding_dim = embedding_dim
        self.device = device or torch.device("cpu")
        self.mapping: Dict[str, int] = {}
        self.embedding: torch.nn.Embedding | None = None

    # start of fit
    def fit(self, source: Union[str, Sequence[str]]) -> None:
        """Learn intention mapping from a list or a NPZ dataset."""
        if isinstance(source, str) and os.path.isfile(source) and source.endswith(".npz"):
            data = np.load(source, allow_pickle=True)
            if "intentions" not in data:
                raise KeyError("NPZ file does not contain 'intentions'")
            raw = data["intentions"]
            if raw.dtype.kind in {"U", "S", "O"}:
                values = [str(x) for x in raw.tolist()]
            else:
                values = [str(x) for x in raw.tolist()]
            vocab = sorted(set(values))
        else:
            vocab = sorted(set(str(x) for x in source))
        self.mapping = {val: idx for idx, val in enumerate(vocab)}
        if self.embedding_dim is not None:
            self.embedding = torch.nn.Embedding(len(self.mapping), self.embedding_dim).to(self.device)
        else:
            self.embedding = None

    def encode(self, intention: str) -> torch.Tensor:
        """Return vector representation for a single intention."""
        if intention not in self.mapping:
            raise KeyError(f"Unknown intention: {intention}")
        idx = self.mapping[intention]
        if self.embedding is None:
            vec = torch.zeros(len(self.mapping), dtype=torch.float32, device=self.device)
            vec[idx] = 1.0
            return vec
        index = torch.tensor([idx], dtype=torch.long, device=self.device)
        return self.embedding(index)[0]

    def encode_batch(self, intentions: Sequence[str]) -> torch.Tensor:
        """Encode a batch of intentions."""
        indices = [self.mapping[i] for i in intentions]
        tensor_idx = torch.tensor(indices, dtype=torch.long, device=self.device)
        if self.embedding is None:
            return torch.nn.functional.one_hot(tensor_idx, num_classes=len(self.mapping)).float()
        return self.embedding(tensor_idx)

    def to_tensor(self) -> torch.Tensor:
        """Return full encoding table as a tensor."""
        if self.embedding is None:
            return torch.eye(len(self.mapping), device=self.device)
        return self.embedding.weight.data.clone()

    def export_mapping(self) -> Dict[str, int]:
        """Export intention to index mapping."""
        return dict(self.mapping)

    def import_mapping(self, mapping: Dict[str, int]) -> None:
        """Load an external mapping."""
        self.mapping = dict(mapping)
        if self.embedding_dim is not None:
            self.embedding = torch.nn.Embedding(len(self.mapping), self.embedding_dim).to(self.device)
        else:
            self.embedding = None

