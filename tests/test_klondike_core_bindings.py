import json
import pytest
import klondike_core as kc

pytestmark = pytest.mark.skipif(kc._core is None, reason="klondike_core extension not available")


def test_new_game():
    assert callable(kc.new_game)
    state = kc.new_game()
    assert isinstance(state, str)
    data = json.loads(state)
    assert "encoded" in data


def test_legal_moves():
    state = kc.new_game()
    encoded = json.loads(state)["encoded"]
    assert callable(kc.legal_moves)
    moves = kc.legal_moves(encoded)
    assert isinstance(moves, list)


def test_play_move():
    state = kc.new_game()
    encoded = json.loads(state)["encoded"]
    move = kc.legal_moves(encoded)[0]
    assert callable(kc.play_move)
    next_state, valid = kc.play_move(state, move)
    assert isinstance(next_state, str)
    assert isinstance(valid, bool)


def test_compute_base_reward_json():
    state = kc.new_game()
    assert callable(kc.compute_base_reward_json)
    reward = kc.compute_base_reward_json(state)
    assert isinstance(reward, float)


def test_move_index_and_move_from_index():
    state = kc.new_game()
    encoded = json.loads(state)["encoded"]
    move = kc.legal_moves(encoded)[0]
    assert callable(kc.move_index)
    idx = kc.move_index(move)
    assert isinstance(idx, int)
    assert callable(kc.move_from_index)
    mv = kc.move_from_index(idx)
    assert mv is None or isinstance(mv, str)
    if mv is not None:
        assert kc.move_index(mv) == idx


def test_shuffle_seed():
    assert callable(kc.shuffle_seed)
    seed = kc.shuffle_seed()
    assert isinstance(seed, int)


def test_solve_klondike():
    state = kc.new_game()
    assert callable(kc.solve_klondike)
    solution = kc.solve_klondike(state)
    assert isinstance(solution, list)


def test_encode_observation():
    state = kc.new_game()
    assert callable(kc.encode_observation)
    obs = kc.encode_observation(state)
    assert isinstance(obs, list)


def test_is_won_and_is_lost():
    state = kc.new_game()
    assert callable(kc.is_won)
    assert kc.is_won(state) is False
    assert callable(kc.is_lost)
    assert kc.is_lost(state) is False


def test_infer_intention():
    state = kc.new_game()
    encoded = json.loads(state)["encoded"]
    move = kc.legal_moves(encoded)[0]
    next_state, _ = kc.play_move(state, move)
    assert callable(kc.infer_intention)
    intention = kc.infer_intention(state, move, next_state)
    assert isinstance(intention, str)
