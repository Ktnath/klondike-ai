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
    )
except Exception as exc:  # pragma: no cover - explicit error
    raise ImportError(
        "Failed to import klondike_core. Build the Rust extension before running."
    ) from exc

from .reward import compute_reward


class KlondikeEnv(gym.Env):
    """Environment wrapping the Rust Klondike engine for RL."""

    def __init__(self) -> None:
        super().__init__()
        self.state: str | None = None
        self.action_space = gym.spaces.Discrete(96)
        # Observation length used by existing training code
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(156,), dtype=np.float32)

    def _encode_state(self) -> np.ndarray:
        obs = []
        if self.state is not None:
            obs = encode_observation(self.state)
        obs = np.array(obs, dtype=np.float32)
        if obs.size < 156:
            obs = np.pad(obs, (0, 156 - obs.size))
        else:
            obs = obs[:156]
        return obs

    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        if new_game is None:
            raise RuntimeError("klondike_core engine not available")
        self.state = new_game()
        return self._encode_state()

    def get_valid_actions(self) -> List[int]:
        """Return currently valid action indices."""
        if self.state is None:
            return []
        moves = legal_moves(self.state)
        return [move_index(m) for m in moves]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Apply an action to the game."""
        assert self.state is not None, "Environment not initialized"
        prev_state = self.state
        move_json = move_from_index(action)
        if move_json is None:
            valid = False
        else:
            self.state, valid = play_move(self.state, move_json)
        next_state = self.state

        done = bool(is_won(next_state)) or len(legal_moves(next_state)) == 0
        reward = compute_reward(prev_state, action, next_state, done)

        obs = self._encode_state()
        info = {"valid": valid}
        return obs, reward, done, info
