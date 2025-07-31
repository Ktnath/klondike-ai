import os
import sys
import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from env.klondike_env import KlondikeEnv
from env.state_utils import get_hidden_cards, count_empty_columns, extract_foundations


def test_state_analysis_utils():
    env = KlondikeEnv()
    env.reset(seed=None)
    state_dict = json.loads(env.state)

    hidden = get_hidden_cards(state_dict)
    assert isinstance(hidden, set)

    empty = count_empty_columns(state_dict)
    assert isinstance(empty, int)
    assert empty >= 0

    foundations = extract_foundations(state_dict)
    assert isinstance(foundations, dict)
    assert len(foundations) == 4
