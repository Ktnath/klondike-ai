#!/usr/bin/env python
"""Automatic checklist for Klondike-AI components."""
from __future__ import annotations

import importlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


TABLE_HEADERS = (
    "PyO3 bindings",
    "Dataset .npz avec intentions",
    "Observation encodée",
    "Modules Python (env, utils)",
    "config.yaml complet",
    "Fichiers scripts présents",
    "Modèle présent",
    "replay_viewer.py",
    "self-play pipeline",
    "Documentation",
    "Test de compilation",
)


class Audit:
    """Collect status of all checks."""

    def __init__(self) -> None:
        self.status: dict[str, bool] = {header: False for header in TABLE_HEADERS}
        handlers = [logging.StreamHandler(sys.stdout)]
        log_dir = Path("logs")
        if log_dir.is_dir():
            handlers.append(
                logging.FileHandler(
                    log_dir / "audit_report.txt", mode="w", encoding="utf-8"
                )
            )
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s: %(message)s", handlers=handlers
        )
        self.log = logging.getLogger(__name__)
        self.modules: dict[str, object] = {}

    def check_modules(self) -> None:
        """Import core modules."""
        mods = ["klondike_core", "env.klondike_env", "intention_utils"]
        success = True
        for name in mods:
            try:
                self.modules[name] = importlib.import_module(name)
            except Exception as exc:  # pragma: no cover - runtime check
                self.log.error("Module %s missing: %s", name, exc)
                success = False
        self.status["Modules Python (env, utils)"] = success

    def check_bindings(self) -> None:
        core = self.modules.get("klondike_core")
        funcs = [
            "new_game",
            "legal_moves",
            "play_move",
            "solve_klondike",
            "encode_state_to_json",
        ]
        success = True
        if core is None:
            success = False
        else:
            for f in funcs:
                if not hasattr(core, f):
                    self.log.error("Missing binding: %s", f)
                    success = False
        self.status["PyO3 bindings"] = success

    def check_encode(self) -> None:
        core = self.modules.get("klondike_core")
        if core is None or not hasattr(core, "encode_observation"):
            self.log.error("encode_observation unavailable")
            self.status["Observation encodée"] = False
            return
        try:
            state = core.new_game()
            obs = core.encode_observation(state)
            self.status["Observation encodée"] = len(obs) == 156
            if len(obs) != 156:
                self.log.error("encode_observation returned %d values", len(obs))
        except Exception as exc:  # pragma: no cover - runtime check
            self.log.error("encode_observation failed: %s", exc)
            self.status["Observation encodée"] = False

    def check_config(self) -> None:
        try:
            with open("config.yaml", "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            needed = {"env", "training", "model", "logging", "intention_embedding"}
            missing = needed - data.keys()
            if missing:
                self.log.error(
                    "Config sections manquantes: %s", ", ".join(sorted(missing))
                )
                self.status["config.yaml complet"] = False
            else:
                self.status["config.yaml complet"] = True
        except FileNotFoundError:
            self.log.error("config.yaml not found")
            self.status["config.yaml complet"] = False

    def check_dataset(self) -> None:
        success = False
        for path in Path(".").rglob("*.npz"):
            try:
                data = np.load(path, allow_pickle=True)
                if "intentions" in data or "intentions_high" in data:
                    success = True
                    break
            except Exception:
                continue
        if not success:
            self.log.error("No .npz dataset with intentions found")
        self.status["Dataset .npz avec intentions"] = success

    def check_scripts(self) -> None:
        scripts = [
            "generate_expert_dataset.py",
            "self_play_generate.py",
            "train/train_dqn.py",
            "train/imitation_learning.py",
        ]
        success = all(Path(s).is_file() for s in scripts)
        if not success:
            self.log.error("Missing required scripts")
        self.status["Fichiers scripts présents"] = success
        try:
            importlib.import_module("train.train_dqn")
            importlib.import_module("train.imitation_learning")
            self.status["self-play pipeline"] = Path("self_play_generate.py").is_file()
        except Exception as exc:
            self.log.error("Failed to import training modules: %s", exc)
            self.status["self-play pipeline"] = False

    def check_model(self) -> None:
        paths = list(Path(".").rglob("*.pt")) + list(Path(".").rglob("*.ckpt"))
        success = bool(paths)
        if not success:
            self.log.error("No trained model (.pt or .ckpt) found")
        self.status["Modèle présent"] = success

    def check_replay_viewer(self) -> None:
        path = Path("tools/replay_viewer.py")
        success = path.is_file()
        if success:
            try:
                importlib.import_module("tools.replay_viewer")
            except Exception as exc:  # pragma: no cover - runtime check
                self.log.error("replay_viewer import failed: %s", exc)
                success = False
        else:
            self.log.error("replay_viewer.py missing")
        self.status["replay_viewer.py"] = success

    def check_docs(self) -> None:
        required = ["README.md", "docs/expert_dataset.md", "config.yaml"]
        success = all(Path(p).is_file() for p in required)
        if not success:
            self.log.error("Missing documentation files")
        self.status["Documentation"] = success

    def check_compile(self) -> None:
        cmd = ["cargo", "check", "--manifest-path", "core/Cargo.toml", "-q"]
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
            )
            self.status["Test de compilation"] = True
        except Exception as exc:  # pragma: no cover - ignore compile errors
            self.log.error("Compilation failed: %s", exc)
            self.status["Test de compilation"] = False

    def print_summary(self) -> None:
        headers = ["Composant", "Statut"]
        print("-" * 47)
        print(f"| {headers[0]:<28} | {headers[1]:^7} |")
        print("|" + "-" * 28 + "|" + "-" * 7 + "|")
        for h in TABLE_HEADERS:
            status = "✅" if self.status.get(h) else "❌"
            if h == "Test de compilation" and not self.status.get(h):
                status = "❌/*"
            print(f"| {h:<28} | {status:^7} |")
        print("-" * 47)


if __name__ == "__main__":
    audit = Audit()
    audit.check_modules()
    audit.check_bindings()
    audit.check_encode()
    audit.check_config()
    audit.check_dataset()
    audit.check_scripts()
    audit.check_model()
    audit.check_replay_viewer()
    audit.check_docs()
    audit.check_compile()
    audit.print_summary()
