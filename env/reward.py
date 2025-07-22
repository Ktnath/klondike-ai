try:
    from klondike_core import compute_base_reward_json
except Exception:
    compute_base_reward_json = None

USE_RUST_REWARD = True


def compute_reward(json_state: str) -> float:
    if compute_base_reward_json is not None and USE_RUST_REWARD:
        try:
            return compute_base_reward_json(json_state)
        except Exception as e:
            import logging
            logging.warning(f"Rust reward error: {e}")
            return -0.02
    return -0.01
