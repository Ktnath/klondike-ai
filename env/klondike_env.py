"""OpenAI Gym environment wrapper for the Klondike game."""
from __future__ import annotations

import numpy as np
import gym
from typing import Tuple, Dict, Any, List

try:
    from klondike_core import Engine, Move
except Exception:  # pragma: no cover - allow import failure in docs
    Engine = None
    Move = None

from .reward import compute_reward


class KlondikeEnv(gym.Env):
    """Environment wrapping the Rust Klondike engine for RL."""

    def __init__(self) -> None:
        super().__init__()
        self.engine: Engine | None = None
        self.state: Any = None
        self.action_space = gym.spaces.Discrete(96)
        # Observation length used by existing training code
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(156,), dtype=np.float32)

    def _encode_state(self) -> np.ndarray:
        obs = [] if self.state is None else self.state.encode_observation()
        obs = np.array(obs, dtype=np.float32)
        if obs.size < 156:
            obs = np.pad(obs, (0, 156 - obs.size))
        else:
            obs = obs[:156]
        return obs

    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        if Engine is None:
            raise RuntimeError("klondike_core engine not available")
        self.engine = Engine()
        self.state = self.engine.get_state()
        return self._encode_state()

    def get_valid_actions(self) -> List[int]:
        """Return currently valid action indices."""
        if self.engine is None:
            return []
        moves = self.engine.get_available_moves()
        return [m.get_move_index() for m in moves]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Apply an action to the game."""
        assert self.engine is not None, "Environment not initialized"
        move = Move.from_move_index(action)
        prev_state = self.state

        valid = False
        if move is not None:
            valid = self.engine.make_move(move)
            if valid:
                self.state = self.engine.get_state()
        next_state = self.state

        done = bool(next_state.is_won()) or len(self.engine.get_available_moves()) == 0
        reward = compute_reward(prev_state, action, next_state, done)

        obs = self._encode_state()
        info = {"valid": valid}
        return obs, reward, done, info
