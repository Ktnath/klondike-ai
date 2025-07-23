import json
from klondike_core import (
    new_game,
    play_move,
    legal_moves,
    compute_base_reward_json,
    encode_state_to_json,
    move_index,
    move_from_index,
    shuffle_seed,
)


def test_core_bindings_basic():
    state = new_game()
    data = json.loads(state)
    encoded = data["encoded"]

    moves = legal_moves(encoded)
    assert isinstance(moves, list)
    assert moves

    next_state, valid = play_move(state, moves[0])
    assert isinstance(next_state, str)
    assert valid

    reward = compute_base_reward_json(next_state)
    assert isinstance(reward, float)

    encoded_json = encode_state_to_json(encoded)
    assert isinstance(encoded_json, str)

    idx = move_index(moves[0])
    assert isinstance(idx, int)
    mv = move_from_index(idx)
    assert mv is None or isinstance(mv, str)

    seed = shuffle_seed()
    assert isinstance(seed, int)
