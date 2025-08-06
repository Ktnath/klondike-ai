import json
from klondike_core import new_game, solve_klondike

def test_solve_klondike_intentions():
    state = new_game()
    solution = solve_klondike(state)
    assert isinstance(solution, list)
    for mv, intention in solution:
        assert isinstance(mv, str)
        assert isinstance(intention, str)
        assert intention.strip() != ""
