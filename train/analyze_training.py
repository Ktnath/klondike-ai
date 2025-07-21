from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


def load_log(path: str) -> pd.DataFrame:
    """Load a training CSV log."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Log file not found: {path}")
    return pd.read_csv(path)


def moving_average(series: pd.Series, window: int = 100) -> pd.Series:
    """Return moving average for a pandas series."""
    return series.rolling(window, min_periods=1).mean()


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Compute basic statistics from the log dataframe."""
    metrics = {
        "episodes": len(df),
        "avg_length": df["length"].mean() if "length" in df.columns else float("nan"),
        "avg_reward": df["reward"].mean() if "reward" in df.columns else float("nan"),
        "win_rate": df["win_rate"].iloc[-1] if "win_rate" in df.columns else float("nan"),
        "final_epsilon": df["epsilon"].iloc[-1] if "epsilon" in df.columns else float("nan"),
        "final_loss": df["loss"].iloc[-1] if "loss" in df.columns else float("nan"),
    }
    return metrics


def plot_metric(
    df: pd.DataFrame,
    column: str,
    output_dir: str,
    compare_df: Optional[pd.DataFrame] = None,
    title: Optional[str] = None,
) -> None:
    if column not in df.columns:
        return
    episodes = df["episode"] if "episode" in df.columns else range(len(df))
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, df[column], label="Run 1")
    if compare_df is not None and column in compare_df.columns:
        cmp_eps = compare_df["episode"] if "episode" in compare_df.columns else range(len(compare_df))
        plt.plot(cmp_eps, compare_df[column], label="Run 2", linestyle="--")
    plt.xlabel("Episode")
    plt.ylabel(column)
    if title:
        plt.title(title)
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{column}.png"))
    plt.close()


def plot_win_rate(
    df: pd.DataFrame,
    output_dir: str,
    compare_df: Optional[pd.DataFrame] = None,
    window: int = 100,
) -> None:
    if "win_rate" not in df.columns:
        return
    episodes = df["episode"] if "episode" in df.columns else range(len(df))
    win_ma = moving_average(df["win_rate"], window)
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, win_ma, label=f"Run 1 MA{window}")
    if compare_df is not None and "win_rate" in compare_df.columns:
        cmp_eps = compare_df["episode"] if "episode" in compare_df.columns else range(len(compare_df))
        cmp_ma = moving_average(compare_df["win_rate"], window)
        plt.plot(cmp_eps, cmp_ma, label=f"Run 2 MA{window}", linestyle="--")
    plt.xlabel("Episode")
    plt.ylabel("Win rate %")
    plt.title("Taux de victoire (moyenne mobile)")
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "win_rate_ma.png"))
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse post-entraînement des logs DQN")
    parser.add_argument(
        "--logfile",
        type=str,
        default="logs/training_log.csv",
        help="Fichier CSV de log à analyser",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports",
        help="Dossier de sortie pour les graphiques",
    )
    parser.add_argument(
        "--compare",
        type=str,
        help="Fichier CSV à comparer",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    try:
        df = load_log(args.logfile)
    except FileNotFoundError as exc:
        logging.error(exc)
        return
    df_compare = load_log(args.compare) if args.compare else None

    metrics = compute_metrics(df)
    logging.info("Episodes: %d", metrics["episodes"])
    if not pd.isna(metrics["avg_length"]):
        logging.info("Longueur moyenne: %.2f", metrics["avg_length"])
    if not pd.isna(metrics["avg_reward"]):
        logging.info("Récompense moyenne: %.2f", metrics["avg_reward"])
    if not pd.isna(metrics["win_rate"]):
        logging.info("Taux de victoire: %.2f %%", metrics["win_rate"])
    if not pd.isna(metrics["final_epsilon"]):
        logging.info("Epsilon final: %.3f", metrics["final_epsilon"])
    if not pd.isna(metrics["final_loss"]):
        logging.info("Perte finale: %.4f", metrics["final_loss"])

    plot_metric(df, "reward", args.output_dir, df_compare, "Récompense par épisode")
    plot_metric(df, "length", args.output_dir, df_compare, "Longueur d'épisode")
    plot_metric(df, "loss", args.output_dir, df_compare, "Perte moyenne")
    plot_metric(df, "epsilon", args.output_dir, df_compare, "Epsilon")
    plot_win_rate(df, args.output_dir, df_compare)

    logging.info("Graphiques sauvegardés dans %s", args.output_dir)


if __name__ == "__main__":
    main()
