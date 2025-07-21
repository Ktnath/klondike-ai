"""Reward function for Klondike DQN agent."""

from typing import Any
import json
import logging

from utils.config import load_config

try:
    from klondike_core import (
        encode_observation,
        foundation_count,
        compute_base_reward_json,
    )
except Exception:  # pragma: no cover - fallback if extension missing
    encode_observation = None
    foundation_count = None
    compute_base_reward_json = None

try:
    _CFG = load_config()
    USE_RUST_REWARD = bool(getattr(_CFG.env, "use_rust_reward", True))
    USE_SHAPING = bool(getattr(_CFG, "reward_shaping", False))
except Exception:  # pragma: no cover - config may be missing
    USE_RUST_REWARD = True
    USE_SHAPING = False


def _count_face_up(state: Any) -> int:
    """Count face up cards using encoded observation if direct access not available."""
    if hasattr(state, "encode_observation"):
        obs = state.encode_observation()
    elif isinstance(state, str) and encode_observation is not None:
        obs = encode_observation(state)
    else:
        obs = []
    count = 0
    for i in range(2, len(obs), 3):
        if obs[i] > 0.5:
            count += 1
    return count


def _foundation_count(state: Any) -> int:
    """Approximate number of cards in foundations using their top ranks."""
    if hasattr(state, "get_foundation_top"):
        count = 0
        for f in range(4):
            try:
                top = state.get_foundation_top(f)
            except AttributeError:
                top = None
            if top is not None:
                try:
                    count += int(top.rank()) + 1
                except AttributeError:
                    pass
        return count
    elif isinstance(state, str) and foundation_count is not None:
        return foundation_count(state)
    return 0


def is_critical_move(prev_state: Any, next_state: Any) -> bool:
    """Return True if the transition reveals or frees important cards."""
    found_diff = _foundation_count(next_state) - _foundation_count(prev_state)
    flips = _count_face_up(next_state) - _count_face_up(prev_state)
    return found_diff > 0 or flips > 0


def compute_reward(state: Any, action: int, next_state: Any, done: bool) -> float:
    """Compute a shaped reward for the transition.

    Parameters
    ----------
    state : GameState
        Current game state before taking the action.
    action : int
        Discrete action index.
    next_state : GameState
        State after applying the action.
    done : bool
        Whether the episode has terminated.
    """
    if done:
        return 10.0 if getattr(next_state, "is_won", lambda: False)() else -1.0

    reward = 0.0

    if compute_base_reward_json is not None and USE_RUST_REWARD:
        try:
            reward = float(compute_base_reward_json(next_state))
        except Exception as exc:  # pragma: no cover - runtime error
            logging.warning("Rust reward failed, falling back to Python: %s", exc)
            reward = 0.0
    elif compute_base_reward_json is None and USE_RUST_REWARD:
        logging.info("klondike_core missing, using Python reward only")

    flips = _count_face_up(next_state) - _count_face_up(state)
    found_diff = _foundation_count(next_state) - _foundation_count(state)

    if USE_SHAPING:
        if flips > 0:
            reward += 0.1 * flips + 0.05
        if found_diff > 0:
            reward += 1.0 * found_diff
        if flips <= 0 and found_diff <= 0:
            reward -= 0.01
    else:
        if flips > 0:
            reward += 0.1 * flips
        if found_diff > 0:
            reward += 1.0 * found_diff
        if flips <= 0 and found_diff <= 0:
            reward -= 0.01

    return float(reward)
