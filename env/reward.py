try:
    from klondike_core import compute_base_reward_json
except Exception:
    compute_base_reward_json = None

from .is_critical_move import is_critical_move

USE_RUST_REWARD = True


def compute_reward(prev_state: str, action: int, next_state: str, done: bool) -> float:
    """Compute reward for a transition."""
    if compute_base_reward_json is not None and USE_RUST_REWARD:
        try:
            return compute_base_reward_json(next_state)
        except Exception as e:  # pragma: no cover - fallback path
            import logging
            logging.warning(f"Rust reward error: {e}")
            try:
                return is_critical_move(prev_state, next_state)
            except Exception as exc:
                logging.warning(f"Fallback reward error: {exc}")
                return -0.02
    return is_critical_move(prev_state, next_state)
