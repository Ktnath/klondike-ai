"""Integration test chaining new_game → legal_moves → solve_klondike."""

import json
import pytest

import klondike_core as kc


def test_new_game_legal_moves_solve_chain():
    if kc._core is None:
        with pytest.raises(NotImplementedError):
            kc.new_game()
        return

    state = kc.new_game()
    encoded = json.loads(state)["encoded"]
    moves = kc.legal_moves(encoded)
    assert isinstance(moves, list) and moves

    solution = kc.solve_klondike(state)
    assert isinstance(solution, list)

