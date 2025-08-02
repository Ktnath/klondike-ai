import json
from typing import List, Tuple, Optional


def new_game() -> str:
    """Return a minimal dummy game state as JSON string."""
    state = {"encoded": [0, 1, 2], "moves": ["m1", "m2"], "step": 0}
    return json.dumps(state)


def legal_moves(encoded) -> List[str]:
    """Return a list of possible moves for the encoded state."""
    return ["m1", "m2"]


def play_move(state_json: str, move: str) -> Tuple[str, bool]:
    """Pretend to play a move and return new state and whether game is done."""
    state = json.loads(state_json)
    state["step"] += 1
    done = state["step"] >= 2
    state["moves"] = [] if done else ["m1", "m2"]
    return json.dumps(state), done


def compute_base_reward_json(state_json: str) -> float:
    """Return a dummy reward."""
    return 0.5


def encode_state_to_json(encoded) -> str:
    return json.dumps({"encoded": encoded})


def move_index(move: str) -> int:
    return 0


def move_from_index(index: int) -> Optional[str]:
    return "m1" if index == 0 else None


# --- Move ↔ index helpers -------------------------------------------------

# Mapping offsets for each move type. Each move is represented by a type
# string followed by a card index (0-51). The overall index space reserves
# 52 contiguous slots for every move type in order to provide a stable
# bijection between move strings and numeric indices.
_MOVE_TYPE_OFFSETS = {
    "DS": 0 * 52,  # Deck → Stack (foundation)
    "PS": 1 * 52,  # Pile → Stack (foundation)
    "DP": 2 * 52,  # Deck → Pile (tableau)
    "SP": 3 * 52,  # Stack → Pile (tableau)
    "R": 4 * 52,   # Reveal from stock
}
_TOTAL_MOVES = 5 * 52


def move_index_py(move: str) -> int:
    """Return the stable index of ``move`` or ``-1`` if invalid.

    ``move`` must have the form ``"TYPE idx"`` where ``TYPE`` is one of the
    keys in ``_MOVE_TYPE_OFFSETS`` and ``idx`` is a card index in ``0..51``.
    The resulting index encodes the move type and the card in a single
    integer value.
    """

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
    return 42


def solve_klondike(state_json: str) -> List[Tuple[str, str]]:
    """Return a dummy solution consisting of one move with an intention."""
    return [("m1", "reveal")]


def encode_observation(state_json: str):
    """Return a dummy observation array."""
    return [0, 1, 2]

def is_won(state_json: str) -> bool:
    return False

def is_lost(state_json: str) -> bool:
    return False

def infer_intention(move: str) -> str:
    return "reveal"
