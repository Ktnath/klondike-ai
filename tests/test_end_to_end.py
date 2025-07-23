import os
import sys
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from env.klondike_env import KlondikeEnv

MAX_STEPS = 20


def test_random_game_runs():
    env = KlondikeEnv()
    env.reset()

    done = False
    step_count = 0
    while not done and step_count < MAX_STEPS:
        actions = env.get_valid_actions()
        assert actions
        action = random.choice(actions)
        obs, reward, done, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        step_count += 1
    assert step_count > 0
