"""Hyperparameter optimization for DQN using Optuna."""
from __future__ import annotations

import argparse
import os
import shutil
import logging
from typing import Any

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import optuna
import yaml
from tqdm import tqdm


# Automatically patched for modular project structure via bootstrap.py
from bootstrap import *

from utils.config import load_config, DotDict
from train.train_dqn import train
from train.evaluate_dqn import evaluate


def objective(trial: optuna.Trial) -> float:
    """Objective function for Optuna trials."""
    config = load_config()

    # Sample hyperparameters
    config.training.learning_rate = trial.suggest_float(
        "learning_rate", 1e-5, 1e-3, log=True
    )
    config.training.gamma = trial.suggest_float("gamma", 0.90, 0.99)
    config.training.batch_size = trial.suggest_int("batch_size", 32, 128)
    config.training.epsilon.decay = trial.suggest_float(
        "epsilon_decay", 0.99, 0.9999
    )

    # Short training for evaluation
    config.training.episodes = 100
    config.logging.enable_logging = False

    log_dir = os.path.join("logs", "optuna")
    os.makedirs(log_dir, exist_ok=True)

    config.logging.log_path = os.path.join(log_dir, f"trial_{trial.number}.csv")
    config.model.save_path = os.path.join(log_dir, f"trial_{trial.number}.pth")

    train(config, force_dim_check=False)

    results = evaluate(
        config.model.save_path, episodes=20, greedy_policy=True, config=config
    )
    win_rate = results["wins"] / results["episodes"] if results["episodes"] else 0.0
    return win_rate


def log_best_trials(study: optuna.Study, trial: optuna.Trial) -> None:
    """Callback to log the top five trials."""
    top = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value,
        reverse=True,
    )[:5]
    logging.info("Top 5 trials so far:")
    for rank, t in enumerate(top, 1):
        logging.info("  %d. Trial %d -> %.4f", rank, t.number, t.value)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Optimize DQN hyperparameters")
    parser.add_argument("--trials", type=int, default=30, help="Number of trials")
    args = parser.parse_args()

    study = optuna.create_study(direction="maximize")

    with tqdm(total=args.trials) as pbar:
        def progress_cb(study: optuna.Study, trial: optuna.Trial) -> None:
            pbar.update(1)
            log_best_trials(study, trial)

        study.optimize(objective, n_trials=args.trials, callbacks=[progress_cb])

    best_params = study.best_trial.params
    with open("best_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(best_params, f)

    best_model_src = os.path.join("logs", "optuna", f"trial_{study.best_trial.number}.pth")
    if os.path.exists(best_model_src):
        os.makedirs("models", exist_ok=True)
        shutil.copy(best_model_src, os.path.join("models", "optuna_best.pth"))


if __name__ == "__main__":
    main()
