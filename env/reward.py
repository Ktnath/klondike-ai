try:
    from klondike_core import compute_base_reward_json
except Exception:
    compute_base_reward_json = None

import json
from .is_critical_move import is_critical_move
try:
    from klondike_core import legal_moves, is_won as core_is_won
except Exception:  # pragma: no cover - if rust not compiled
    legal_moves = None
    core_is_won = None

USE_RUST_REWARD = True


def compute_reward(prev_state: str, action: int, next_state: str, done: bool) -> float:
    """Compute reward for a transition."""
    if compute_base_reward_json is not None and USE_RUST_REWARD:
        try:
            base = compute_base_reward_json(next_state)
        except Exception as e:  # pragma: no cover - fallback path
            import logging
            logging.warning(f"Rust reward error: {e}")
            try:
                base = is_critical_move(prev_state, next_state)
            except Exception as exc:  # pragma: no cover - parsing failure
                logging.warning(f"Fallback reward error: {exc}")
                base = -0.02
    else:
        base = is_critical_move(prev_state, next_state)

    bonus = 0.0
    try:
        p = json.loads(prev_state)
        n = json.loads(next_state)
    except Exception:  # pragma: no cover - invalid json
        return float(base)

    # victory bonus
    encoded = n.get("encoded", next_state)
    if core_is_won is not None:
        try:
            if core_is_won(encoded):
                bonus += 10.0
        except Exception:  # pragma: no cover
            pass

    # penalty if no actions available
    if legal_moves is not None:
        try:
            if len(legal_moves(encoded)) == 0 and not bonus:
                bonus -= 1.0
        except Exception:  # pragma: no cover
            pass

    # optional bonus for card flipped or moved to foundation
    try:
        prev_found = sum(len(f) for f in p.get("foundations", []))
        next_found = sum(len(f) for f in n.get("foundations", []))
        diff_found = max(0, next_found - prev_found)
        bonus += diff_found * 1.0
        prev_down = [c.get("face_down", 0) for c in p.get("tableau", [])]
        next_down = [c.get("face_down", 0) for c in n.get("tableau", [])]
        turned = sum(max(0, b - a) for b, a in zip(prev_down, next_down))
        bonus += turned * 1.0
    except Exception:  # pragma: no cover
        pass

    return float(base + bonus)
