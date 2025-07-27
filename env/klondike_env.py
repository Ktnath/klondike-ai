"""OpenAI Gym environment wrapper for the Klondike game."""
from __future__ import annotations

import numpy as np
import gym
from typing import Tuple, Dict, Any, List

try:
    from klondike_core import (
        new_game,
        legal_moves,
        play_move,
        encode_observation,
        move_from_index,
        move_index,
        is_won,
        is_lost,
    )
except Exception as exc:  # pragma: no cover - explicit error
    raise ImportError(
        "Failed to import klondike_core. Build the Rust extension before running."
    ) from exc

from .reward import compute_reward
from .state_utils import count_empty_columns
import json


def _infer_intention(before: str, mv: str, after: str) -> str:
    """Infer intention label from two JSON states and a move."""
    try:
        before_dict = json.loads(before)
        after_dict = json.loads(after)
    except Exception:
        return ""
    if mv.startswith("R"):
        return "reveal"
    try:
        from klondike_core import foundation_count
        if foundation_count(after) > foundation_count(before):
            return "foundation"
    except Exception:
        pass
    parts = mv.split()
    mv_type = parts[0] if parts else ""
    idx = int(parts[1]) if len(parts) > 1 else 0
    if mv_type in {"DP", "SP"}:
        if idx // 4 == 12 and count_empty_columns(before_dict) > count_empty_columns(after_dict):
            return "king_to_empty"
        return "stack_move"
    if mv_type in {"DS", "PS"}:
        return "foundation"
    return "stack_move"


class KlondikeEnv(gym.Env):
    """Environment wrapping the Rust Klondike engine for RL."""

    def __init__(self, seed: str | None = None, use_intentions: bool = True) -> None:
        super().__init__()
        self.state: str | None = None
        self.seed: str | None = seed
        self.use_intentions = use_intentions
        self.action_space = gym.spaces.Discrete(96)
        dim = 156 + 4 if self.use_intentions else 156
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(dim,), dtype=np.float32)

    def _intention_to_vec(self, intention: str | None) -> np.ndarray:
        mapping = {
            "reveal": 0,
            "foundation": 1,
            "stack_move": 2,
            "king_to_empty": 3,
        }
        vec = np.zeros(4, dtype=np.float32)
        if not intention:
            return vec
        key = str(intention).strip().lower().replace(" ", "_")
        for name, idx in mapping.items():
            if name in key:
                vec[idx] = 1.0
                break
        return vec

    def _encode_state(self, intention: str | None = None) -> np.ndarray:
        obs = []
        if self.state is not None:
            obs = encode_observation(self.state)
        obs = np.array(obs, dtype=np.float32)
        if obs.size < 156:
            obs = np.pad(obs, (0, 156 - obs.size))
        else:
            obs = obs[:156]
        if self.use_intentions:
            obs = np.concatenate([obs, self._intention_to_vec(intention)])
        return obs

    def reset(self, seed: str | None = None) -> np.ndarray:
        """Reset environment and return initial observation.

        Parameters
        ----------
        seed:
            Optional seed to create a deterministic game. If provided, it is
            forwarded to the Rust engine when available.
        """
        if new_game is None:
            raise RuntimeError("klondike_core engine not available")
        if seed is None:
            seed = self.seed
        self.seed = seed
        if seed is not None:
            try:
                self.state = new_game(seed)
            except TypeError:
                # Fallback if new_game does not accept a seed
                self.state = new_game()
        else:
            self.state = new_game()
        return self._encode_state()

    def _encoded(self, state: str) -> str:
        """Extract the encoded state string from the JSON representation."""
        try:
            import json

            return json.loads(state)["encoded"]
        except Exception:
            return state

    def get_valid_actions(self) -> List[int]:
        """Return currently valid action indices."""
        if self.state is None:
            return []
        encoded = self._encoded(self.state)
        moves = legal_moves(encoded)
        return [move_index(m) for m in moves]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Apply an action to the game."""
        assert self.state is not None, "Environment not initialized"

        prev = self.state
        move = move_from_index(action)
        if move is None:
            valid = False
            intention = ""
        else:
            self.state, valid = play_move(self.state, move)
            intention = _infer_intention(prev, move, self.state) if self.use_intentions else ""

        encoded = self._encoded(self.state)
        done = bool(is_won(encoded)) or bool(is_lost(encoded))
        reward = compute_reward(self.state, move or "", done=done)
        info: Dict[str, Any] = {"valid": bool(valid), "intention": intention}
        obs = self._encode_state(intention)

        return obs, reward, done, info
