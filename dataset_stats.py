#!/usr/bin/env python3
"""Utility script to inspect expert datasets.

This script loads a dataset produced by ``generate_expert_dataset.py`` and
prints a number of helpful statistics.  Both ``.npz`` and ``.csv`` files are
supported.  When intentions are present, their distribution can be analysed and
optionally plotted.

The goal is to quickly sanity–check a dataset before using it for training.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# Optional imports for plotting.  These are loaded lazily so the script works
# even if the libraries are missing.
try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:  # pragma: no cover - plotting is optional
    plt = None
    sns = None


def _warn_unbalanced(counter: Counter, total: int) -> None:
    """Warn the user if an intention distribution looks unbalanced.

    A simple heuristic is used: any intention representing less than 5% of the
    dataset triggers a warning.  This is merely informative and does not stop
    execution.
    """

    for key, count in counter.items():
        ratio = count / total
        if ratio < 0.05:
            print(f"⚠️  Intent '{key}' is underrepresented: {ratio:.2%}")


def _analyse_intentions(intentions: Iterable, top: int, plot: bool) -> None:
    """Display statistics about intentions.

    Parameters
    ----------
    intentions:
        Iterable containing the intentions associated with each sample.
    top:
        Number of most common intentions to display.
    plot:
        Whether to show a histogram of the intention distribution using
        ``matplotlib``/``seaborn`` if available.
    """

    intentions_list = list(intentions)
    counter = Counter(intentions_list)
    print(f"Unique intentions: {len(counter)}")
    for intent, count in counter.most_common(top):
        print(f"{intent!r}: {count}")

    # Detect missing intentions
    missing = [i for i, x in enumerate(intentions_list) if x in (None, "")]
    if missing:
        print(f"⚠️  Missing intentions for {len(missing)} samples")

    _warn_unbalanced(counter, len(intentions_list))

    if plot and plt is not None and sns is not None:  # pragma: no cover - visual
        plt.figure(figsize=(8, 4))
        sns.countplot(x=intentions_list)
        plt.title("Intention distribution")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
    elif plot:
        print("Plotting requested but matplotlib/seaborn are not available.")


def analyse_npz(path: str, top: int, plot: bool) -> None:
    """Analyse a dataset stored as ``.npz``."""

    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:  # pragma: no cover - depends on user input
        print(f"❌ Failed to load NPZ file '{path}': {exc}")
        return

    observations = data.get("observations")
    # Some datasets might use 'targets' while others use 'actions'
    targets = data.get("targets")
    if targets is None:
        targets = data.get("actions")
    intentions = data.get("intentions")
    rewards = data.get("rewards")

    if observations is None or targets is None:
        print("⚠️  Dataset is missing required 'observations' or 'targets/actions' arrays")
        return

    print(f"Number of samples: {len(observations)}")
    print(f"Observations shape: {observations.shape}")
    print(f"Targets shape: {targets.shape}")

    # Consistency checks
    if len(observations) != len(targets):
        print(
            f"⚠️  Observations and targets length mismatch: {len(observations)} != {len(targets)}"
        )

    empty = [i for i, obs in enumerate(observations) if hasattr(obs, "__len__") and len(obs) == 0]
    if empty:
        print(f"⚠️  Found {len(empty)} empty observations")

    if intentions is not None:
        print("Intentions field detected")
        _analyse_intentions(intentions, top, plot)
    else:
        print("No 'intentions' field found")

    if rewards is not None:
        print(
            "Rewards stats: min={:.3f}, max={:.3f}, mean={:.3f}".format(
                float(np.min(rewards)), float(np.max(rewards)), float(np.mean(rewards))
            )
        )


def analyse_csv(path: str, top: int, plot: bool) -> None:
    """Analyse a dataset stored as ``.csv``."""

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - depends on user input
        print(f"❌ Failed to load CSV file '{path}': {exc}")
        return

    print(f"Number of samples: {len(df)}")
    if "observation" in df.columns:
        print("Column 'observation' found")
    if "action" in df.columns:
        print("Column 'action' found")
    if "target" in df.columns:
        print("Column 'target' found")

    # Empty rows detection
    empty_rows = df[df.isna().any(axis=1)]
    if not empty_rows.empty:
        print(f"⚠️  Found {len(empty_rows)} rows with missing values")

    if "intention" in df.columns:
        intentions = df["intention"].dropna().tolist()
        print("Intentions column detected")
        _analyse_intentions(intentions, top, plot)
    else:
        print("No 'intention' column found")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Analyse a dataset of expert moves")
    parser.add_argument(
        "--input",
        type=str,
        default="data/expert_dataset.npz",
        help="Path to dataset (.npz or .csv)",
    )
    parser.add_argument(
        "--top", type=int, default=10, help="Number of top intentions to display"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Plot intention histogram if possible"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not os.path.exists(args.input):
        print(f"❌ File '{args.input}' does not exist")
        return

    ext = os.path.splitext(args.input)[1].lower()
    if ext == ".npz":
        analyse_npz(args.input, args.top, args.plot)
    elif ext == ".csv":
        analyse_csv(args.input, args.top, args.plot)
    else:
        print(f"❌ Unsupported file extension: {ext}")


if __name__ == "__main__":
    main()
