import json
import os
import numpy as np
import pandas as pd
import pytest

import dataset_stats


def test_analyse_npz_with_intentions_and_rewards(tmp_path, capsys):
    observations = np.zeros((2, 3))
    targets = np.ones((2,))
    intentions = np.array(["a", "b"], dtype=object)
    rewards = np.array([0.1, 0.2])
    path = tmp_path / "data.npz"
    np.savez(path, observations=observations, targets=targets,
             intentions=intentions, rewards=rewards)
    dataset_stats.analyse_npz(str(path), top=5, plot=False)
    out = capsys.readouterr().out
    assert "Number of samples: 2" in out
    assert "Intentions field detected" in out
    assert "Rewards stats: min=0.100" in out


def test_analyse_npz_missing_required_fields(tmp_path, capsys):
    observations = np.zeros((2, 3))
    path = tmp_path / "bad.npz"
    np.savez(path, observations=observations)
    dataset_stats.analyse_npz(str(path), top=5, plot=False)
    out = capsys.readouterr().out
    assert "missing required 'observations' or 'targets/actions'" in out


def test_analyse_npz_length_mismatch(tmp_path, capsys):
    observations = np.zeros((3, 3))
    targets = np.ones((2,))
    path = tmp_path / "mismatch.npz"
    np.savez(path, observations=observations, targets=targets)
    dataset_stats.analyse_npz(str(path), top=5, plot=False)
    out = capsys.readouterr().out
    assert "length mismatch" in out


def test_analyse_csv_basic(tmp_path, capsys):
    df = pd.DataFrame({
        "observation": ["o1", "o2"],
        "action": [0, 1],
        "intention": ["a", "b"],
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    dataset_stats.analyse_csv(str(path), top=5, plot=False)
    out = capsys.readouterr().out
    assert "Number of samples: 2" in out
    assert "Intentions column detected" in out


def test_analyse_csv_missing_values_and_no_intention(tmp_path, capsys):
    df = pd.DataFrame({
        "observation": ["o1", None],
        "action": [0, 1],
    })
    path = tmp_path / "missing.csv"
    df.to_csv(path, index=False)
    dataset_stats.analyse_csv(str(path), top=5, plot=False)
    out = capsys.readouterr().out
    assert "Found 1 rows with missing values" in out
    assert "No 'intention' column found" in out


def test_analyse_intentions_unbalanced_and_missing(capsys):
    intentions = ["a"] * 21 + ["b", None, ""]
    dataset_stats._analyse_intentions(intentions, top=5, plot=False)
    out = capsys.readouterr().out
    assert "Missing intentions for 2 samples" in out
    assert "Intent 'b' is underrepresented" in out


def test_main_nonexistent_file(tmp_path, capsys):
    missing = tmp_path / "nope.npz"
    dataset_stats.main(["--input", str(missing)])
    out = capsys.readouterr().out
    assert f"File '{missing}' does not exist" in out


def test_main_unsupported_extension(tmp_path, capsys):
    txt = tmp_path / "data.txt"
    txt.write_text("dummy")
    dataset_stats.main(["--input", str(txt)])
    out = capsys.readouterr().out
    assert "Unsupported file extension: .txt" in out


def test_main_default_argument(monkeypatch, tmp_path, capsys):
    # Create a tiny dataset to avoid dependency on repo files
    real = tmp_path / "expert_dataset.npz"
    np.savez(real, observations=np.zeros((1, 2)), targets=np.zeros((1,)))

    # Pretend the default path exists and redirect analysis to our temp file
    monkeypatch.setattr(dataset_stats.os.path, "exists", lambda p: p == "data/expert_dataset.npz")
    original = dataset_stats.analyse_npz

    def fake_analyse(path, top, plot):
        assert path == "data/expert_dataset.npz"
        original(str(real), top, plot)

    monkeypatch.setattr(dataset_stats, "analyse_npz", fake_analyse)

    dataset_stats.main([])
    out = capsys.readouterr().out
    assert "Number of samples: 1" in out
