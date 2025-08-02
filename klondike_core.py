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
