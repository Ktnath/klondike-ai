"""Plot training metrics from a CSV/JSON/Pickle log file."""
from __future__ import annotations

import argparse
import os
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_logs(path: str) -> pd.DataFrame:
    """Load training logs from CSV, JSON or pickle."""
    ext = Path(path).suffix
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".json", ".js"}:
        return pd.read_json(path)
    if ext in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported log format: {ext}")


def plot_metrics(df: pd.DataFrame, output: str) -> None:
    """Plot reward and loss curves and optionally win rate."""
    plt.figure(figsize=(10, 6))

    if "episode" in df.columns:
        x = df["episode"]
    else:
        x = range(len(df))

    if "reward" in df.columns:
        plt.plot(x, df["reward"], label="Episode Reward", alpha=0.7)

    if "loss" in df.columns:
        plt.plot(x, df["loss"], label="Loss", alpha=0.7)

    if "win_rate" in df.columns:
        plt.plot(x, df["win_rate"], label="Win Rate", alpha=0.7)

    plt.xlabel("Episode")
    if "win_rate" in df.columns:
        plt.title("Win Rate over Episodes")
    else:
        plt.title("Training metrics")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output)
    plt.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Plot DQN training results")
    parser.add_argument(
        "--log", type=str, default="results/train_log.csv", help="Path to log file"
    )
    parser.add_argument(
        "--output", type=str, default="plots/training_plot.png", help="Output PNG"
    )
    args = parser.parse_args()

    df = load_logs(args.log)
    plot_metrics(df, args.output)
    logging.info("Plot saved to %s", args.output)


if __name__ == "__main__":
    main()
