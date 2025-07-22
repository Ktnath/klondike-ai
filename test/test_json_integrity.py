import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pytest
from klondike_core import new_game, play_move, compute_base_reward_json


def test_state_integrity():
    state = new_game()
    for _ in range(10):
        reward = compute_base_reward_json(state)
        assert isinstance(reward, float)
        assert 0 <= reward <= 1

        import json
        moves = json.loads(state).get("moves", [])
        if not moves:
            break

        state, done = play_move(state, moves[0])
        assert isinstance(state, str)
        if done:
            break
