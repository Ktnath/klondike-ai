import json
from klondike_core import new_game, solve_klondike

def test_solve_klondike_intentions():
    state = new_game()
    solution = solve_klondike(state)
    assert isinstance(solution, list)
    allowed = {"reveal", "foundation", "stack_move", "king_to_empty"}
    for mv, intention in solution:
        assert isinstance(mv, str)
        assert intention in allowed
