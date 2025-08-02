"""Type hints for the Python wrapper around the Rust `klondike_core` extension."""

from __future__ import annotations

from typing import List, Optional, Tuple


def new_game(seed: Optional[str] = ...) -> str: ...


def legal_moves(encoded: str) -> List[str]: ...


def play_move(state_json: str, move: str) -> Tuple[str, bool]: ...


def compute_base_reward_json(state_json: str) -> float: ...


def encode_state_to_json(encoded: str) -> str: ...


def move_index(move: str) -> int: ...


def move_from_index(index: int) -> Optional[str]: ...


def move_index_py(move: str) -> int: ...


def move_from_index_py(index: int) -> Optional[str]: ...


def shuffle_seed() -> int: ...


def solve_klondike(state_json: str) -> List[Tuple[str, str]]: ...


def encode_observation(state_json: str) -> List[float]: ...


def is_won(state_json: str) -> bool: ...


def is_lost(state_json: str) -> bool: ...


def infer_intention(before: str, move: str, after: str) -> str: ...


__all__: list[str]

