# 📄 FICHIER : tests/test_move_index.py
# ✅ OBJECTIF : Vérifier que move_index_py et move_from_index_py assurent un mapping fiable des coups

import pytest
from klondike_core import move_index_py, move_from_index_py

# Liste de quelques coups valides représentatifs
valid_moves = [
    "DS 0", "DS 12",       # moves vers les fondations
    "DP 8", "DP 25",       # stack move
    "R 4", "R 31",         # draw from stock
    "PS 0", "PS 11",       # foundation to tableau
    "SP 12", "SP 27",      # king to empty
]

# 🧪 Test que chaque coup valide retourne un index ≥ 0
@pytest.mark.parametrize("move", valid_moves)
def test_move_index_is_valid(move):
    idx = move_index_py(move)
    assert isinstance(idx, int)
    assert idx >= 0, f"Move '{move}' returned invalid index {idx}"

# 🧪 Test que le round-trip move → index → move est stable
@pytest.mark.parametrize("move", valid_moves)
def test_move_round_trip(move):
    idx = move_index_py(move)
    recovered = move_from_index_py(idx)
    assert recovered == move, f"Round-trip failed: {move} → {idx} → {recovered}"

# 🧪 Test d’un coup invalide
def test_invalid_move_returns_minus1():
    assert move_index_py("INVALID") == -1

# 🧪 Test d’un index invalide
def test_invalid_index_returns_none():
    assert move_from_index_py(-42) is None
    assert move_from_index_py(9999) is None
