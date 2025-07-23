import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from env.klondike_env import KlondikeEnv
from env.state_utils import get_hidden_cards, count_empty_columns, extract_foundations


def test_state_analysis_utils():
    env = KlondikeEnv()
    env.reset()
    state_dict = json.loads(env.state)

    hidden = get_hidden_cards(state_dict)
    assert isinstance(hidden, set)

    empty = count_empty_columns(state_dict)
    assert isinstance(empty, int)
    assert empty >= 0

    foundations = extract_foundations(state_dict)
    assert isinstance(foundations, dict)
    assert len(foundations) == 4
