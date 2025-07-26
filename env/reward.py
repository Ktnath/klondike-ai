try:
    from klondike_core import compute_base_reward_json
except Exception:
    compute_base_reward_json = None

try:
    from klondike_core import is_won as core_is_won, is_lost as core_is_lost
except Exception:  # pragma: no cover - if rust not compiled
    core_is_won = None
    core_is_lost = None

USE_RUST_REWARD = True


def compute_reward(state: str, move: str, *, done: bool) -> float:
    """
    Compute reward for a transition.

    Parameters
    ----------
    state:
        Game state after the move (JSON string).
    move:
        Move just played, JSON representation.
    done:
        Whether the game is finished after this move.
    """
    if compute_base_reward_json is not None and USE_RUST_REWARD:
        try:
            base = compute_base_reward_json(state)
        except Exception as e:  # pragma: no cover - fallback path
            import logging
            logging.warning(f"Rust reward error: {e}")
            base = 0.0
    else:
        base = 0.0

    bonus = 0.0

    if done:
        try:
            if core_is_won is not None and core_is_won(state):
                bonus += 10.0
            elif core_is_lost is not None and core_is_lost(state):
                bonus -= 1.0
        except Exception:  # pragma: no cover
            pass

    return float(base + bonus)
