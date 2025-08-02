import importlib
import subprocess
import sys
import os
from pathlib import Path

import numpy as np


def _create_dummy_dataset(path: Path) -> None:
    """Create a small valid NPZ dataset with intentions."""
    # Deterministic simple observations
    obs = np.zeros((5, 156), dtype=np.float32)
    for i in range(5):
        obs[i, i] = 1.0
    actions = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    intentions = np.array([0, 1, 2, 3, 0], dtype=np.int64)
    np.savez(path, observations=obs, actions=actions, intentions=intentions)


def test_imitation_pipeline_minimal(tmp_path):
    dataset = tmp_path / "mini_dataset.npz"
    _create_dummy_dataset(dataset)

    # Ensure module imports without error
    importlib.import_module("train.imitation_learning")

    cmd = [
        sys.executable,
        "-m",
        "train.imitation_learning",
        "--dataset",
        str(dataset),
        "--epochs",
        "1",
        "--use_intentions",
        "--output_path",
        str(tmp_path / "model.pt"),
    ]
    subprocess.run(cmd, check=True, env={**os.environ, "MPLBACKEND": "Agg"})


def test_dqn_pipeline_minimal(tmp_path):
    dataset = tmp_path / "mini_dataset.npz"
    _create_dummy_dataset(dataset)

    importlib.import_module("train.train_dqn")

    cmd = [
        sys.executable,
        "-m",
        "train.train_dqn",
        "--dataset",
        str(dataset),
        "--epochs",
        "1",
        "--use_intentions",
        "--model_path",
        str(tmp_path / "dqn.pth"),
        "--log_path",
        str(tmp_path / "log.csv"),
    ]
    subprocess.run(cmd, check=True, env={**os.environ, "MPLBACKEND": "Agg"})
