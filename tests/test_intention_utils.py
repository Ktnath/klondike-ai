import logging
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from intention_utils import simplify_and_filter


def test_simplify_and_filter(caplog):
    config = {
        "intention_embedding": {
            "filter_list": ["move_useless_card"],
            "replacements": {
                "reveal_from_stack": "reveal",
                "move_king_to_empty": "empty_col_management",
            },
        }
    }

    raw = [
        "reveal_from_stack",
        "move_useless_card",
        "move_king_to_empty",
        "deplacer quelque chose",
    ]

    with caplog.at_level(logging.INFO):
        result = simplify_and_filter(raw, config)

    assert result == [
        "reveal",
        "empty_col_management",
        "Déplacer vers pile",
    ]
    # ensure logs mention removed and renamed intentions
    assert any("Renamed intention" in msg for msg in caplog.text.splitlines())
    assert any("Filtered intention" in msg for msg in caplog.text.splitlines())

