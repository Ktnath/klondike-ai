"""Python wrapper for the Rust ``klondike_core`` extension.

This module attempts to load the compiled Rust extension of the same name
from the Python path.  If the extension cannot be found the exposed
functions raise :class:`NotImplementedError` to signal that the engine is
not available.

The pure-Python helpers ``move_index_py`` and ``move_from_index_py`` are
kept for convenience and testing purposes.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import importlib
import os
import sys


def _load_rust_module():
    """Load the real ``klondike_core`` extension if it exists on ``sys.path``.

    The current directory (which contains this shim) is temporarily removed
    from ``sys.path`` so that Python can discover an actual compiled module
    with the same name elsewhere (typically in ``site-packages``).
    """

    module_name = "klondike_core"
    current_dir = os.path.abspath(os.path.dirname(__file__))
    original_path = list(sys.path)
    original_module = sys.modules.get(module_name)

    try:
        if module_name in sys.modules:
            del sys.modules[module_name]
        sys.path = [
            p
            for p in sys.path
            if os.path.abspath(p or ".") != current_dir
        ]
        return importlib.import_module(module_name)
    except Exception:
        return None
    finally:
        sys.path[:] = original_path
        if original_module is not None:
            sys.modules[module_name] = original_module


_core = _load_rust_module()


def _missing(*_args, **_kwargs):  # pragma: no cover - defensive utility
    raise NotImplementedError(
        "Rust extension 'klondike_core' is required for this function"
    )


def new_game(seed: Optional[str] = None) -> str:
    """Create a new game and return its JSON representation."""

    if _core is None:
        _missing()
    if seed is not None:
        return _core.new_game(seed)
    return _core.new_game()


def legal_moves(encoded: str) -> List[str]:
    """Return the list of legal moves for an encoded state."""

    if _core is None:
        _missing()
    return _core.legal_moves(encoded)


def play_move(state_json: str, move: str) -> Tuple[str, bool]:
    """Apply ``move`` on ``state_json`` and return the next state and validity."""

    if _core is None:
        _missing()
    return _core.play_move(state_json, move)


def compute_base_reward_json(state_json: str) -> float:
    """Compute the base reward for ``state_json``."""

    if _core is None:
        _missing()
    return float(_core.compute_base_reward_json(state_json))


def encode_state_to_json(encoded: str) -> str:
    """Return the JSON representation of an encoded state."""

    if _core is None:
        _missing()
    return _core.encode_state_to_json(encoded)


def move_index(move: str) -> int:
    """Return a stable numeric index for ``move``."""

    if _core is None:
        _missing()
    return int(_core.move_index(move))


def move_from_index(index: int) -> Optional[str]:
    """Recover the move string from its numeric ``index``."""

    if _core is None:
        _missing()
    return _core.move_from_index(index)


# --- Move ↔ index helpers -------------------------------------------------

# Mapping offsets for each move type. Each move is represented by a type
# string followed by a card index (0-51). The overall index space reserves
# 52 contiguous slots for every move type in order to provide a stable
# bijection between move strings and numeric indices.  These helpers are kept
# as pure Python utilities used in tests.
_MOVE_TYPE_OFFSETS = {
    "DS": 0 * 52,  # Deck → Stack (foundation)
    "PS": 1 * 52,  # Pile → Stack (foundation)
    "DP": 2 * 52,  # Deck → Pile (tableau)
    "SP": 3 * 52,  # Stack → Pile (tableau)
    "R": 4 * 52,  # Reveal from stock
}
_TOTAL_MOVES = 5 * 52


def move_index_py(move: str) -> int:
    """Return the stable index of ``move`` or ``-1`` if invalid."""

    try:
        mv_type, idx_str = move.split()
        card_idx = int(idx_str)
    except ValueError:
        return -1

    offset = _MOVE_TYPE_OFFSETS.get(mv_type)
    if offset is None or not (0 <= card_idx < 52):
        return -1

    return offset + card_idx


def move_from_index_py(index: int) -> Optional[str]:
    """Recover the move string from ``index`` or ``None`` if invalid."""

    if not (0 <= index < _TOTAL_MOVES):
        return None

    mv_type_idx, card_idx = divmod(index, 52)
    mv_types = list(_MOVE_TYPE_OFFSETS.keys())
    mv_type = mv_types[mv_type_idx]
    return f"{mv_type} {card_idx}"


def shuffle_seed() -> int:
    """Return the seed used by the shuffler."""

    if _core is None:
        _missing()
    return int(_core.shuffle_seed())


def solve_klondike(state_json: str) -> List[Tuple[str, str]]:
    """Solve the game described by ``state_json``.

    The result is a list of ``(move, intention)`` tuples.
    """

    if _core is None:
        _missing()
    return _core.solve_klondike(state_json)


def encode_observation(state_json: str) -> List[float]:
    """Encode ``state_json`` into an observation vector."""

    if _core is None:
        _missing()
    return _core.encode_observation(state_json)


def is_won(state_json: str) -> bool:
    """Return ``True`` if the game is won."""

    if _core is None:
        _missing()
    return bool(_core.is_won(state_json))


def is_lost(state_json: str) -> bool:
    """Return ``True`` if the game is lost."""

    if _core is None:
        _missing()
    return bool(_core.is_lost(state_json))


def infer_intention(before: str, move: str, after: str) -> str:
    """Infer the intention of ``move`` using ``before`` and ``after`` states."""

    if _core is None:
        _missing()
    return _core.infer_intention(before, move, after)


__all__ = [
    "new_game",
    "legal_moves",
    "play_move",
    "compute_base_reward_json",
    "encode_state_to_json",
    "move_index",
    "move_from_index",
    "move_index_py",
    "move_from_index_py",
    "shuffle_seed",
    "solve_klondike",
    "encode_observation",
    "is_won",
    "is_lost",
    "infer_intention",
]

