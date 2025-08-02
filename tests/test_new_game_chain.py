"""Simple integration test chaining new_game → legal_moves → solve_klondike."""

import json

from klondike_core import new_game, legal_moves, solve_klondike


def test_new_game_legal_moves_solve_chain():
    state = new_game()
    encoded = json.loads(state)["encoded"]
    moves = legal_moves(encoded)
    assert isinstance(moves, list) and moves

    solution = solve_klondike(state)
    assert isinstance(solution, list)

