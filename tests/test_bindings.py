"""Basic smoke tests for the ``klondike_core`` Python shims.

Each function should either forward to the Rust extension (if available) or
raise :class:`NotImplementedError` when the extension is missing.  These tests
exercise both behaviours.
"""

import json

import pytest

import klondike_core as kc


HAS_CORE = kc._core is not None


def test_new_game():
    if HAS_CORE:
        state = kc.new_game()
        assert isinstance(state, str)
        assert "encoded" in json.loads(state)
    else:
        with pytest.raises(NotImplementedError):
            kc.new_game()


def test_legal_moves():
    if HAS_CORE:
        state = kc.new_game()
        encoded = json.loads(state)["encoded"]
        moves = kc.legal_moves(encoded)
        assert isinstance(moves, list)
    else:
        with pytest.raises(NotImplementedError):
            kc.legal_moves("")


def test_play_move():
    if HAS_CORE:
        state = kc.new_game()
        encoded = json.loads(state)["encoded"]
        move = kc.legal_moves(encoded)[0]
        next_state, valid = kc.play_move(state, move)
        assert isinstance(next_state, str)
        assert isinstance(valid, bool)
    else:
        with pytest.raises(NotImplementedError):
            kc.play_move("{}", "DS 0")


def test_compute_base_reward_json():
    if HAS_CORE:
        state = kc.new_game()
        reward = kc.compute_base_reward_json(state)
        assert isinstance(reward, float)
    else:
        with pytest.raises(NotImplementedError):
            kc.compute_base_reward_json("{}")


def test_move_index():
    if HAS_CORE:
        state = kc.new_game()
        encoded = json.loads(state)["encoded"]
        move = kc.legal_moves(encoded)[0]
        idx = kc.move_index(move)
        assert isinstance(idx, int)
    else:
        with pytest.raises(NotImplementedError):
            kc.move_index("DS 0")


def test_move_from_index():
    if HAS_CORE:
        state = kc.new_game()
        encoded = json.loads(state)["encoded"]
        move = kc.legal_moves(encoded)[0]
        idx = kc.move_index(move)
        mv = kc.move_from_index(idx)
        assert mv is None or isinstance(mv, str)
    else:
        with pytest.raises(NotImplementedError):
            kc.move_from_index(0)


def test_shuffle_seed():
    if HAS_CORE:
        seed = kc.shuffle_seed()
        assert isinstance(seed, int)
    else:
        with pytest.raises(NotImplementedError):
            kc.shuffle_seed()


def test_solve_klondike():
    if HAS_CORE:
        state = kc.new_game()
        solution = kc.solve_klondike(state)
        assert isinstance(solution, list)
    else:
        with pytest.raises(NotImplementedError):
            kc.solve_klondike("{}")


def test_encode_observation():
    if HAS_CORE:
        state = kc.new_game()
        obs = kc.encode_observation(state)
        assert isinstance(obs, list)
    else:
        with pytest.raises(NotImplementedError):
            kc.encode_observation("{}")


def test_is_won():
    if HAS_CORE:
        state = kc.new_game()
        assert kc.is_won(state) is False
    else:
        with pytest.raises(NotImplementedError):
            kc.is_won("{}")


def test_is_lost():
    if HAS_CORE:
        state = kc.new_game()
        assert kc.is_lost(state) is False
    else:
        with pytest.raises(NotImplementedError):
            kc.is_lost("{}")


def test_infer_intention():
    if HAS_CORE:
        state = kc.new_game()
        encoded = json.loads(state)["encoded"]
        move = kc.legal_moves(encoded)[0]
        next_state, _ = kc.play_move(state, move)
        intention = kc.infer_intention(state, move, next_state)
        assert isinstance(intention, str)
    else:
        with pytest.raises(NotImplementedError):
            kc.infer_intention("{}", "DS 0", "{}")

