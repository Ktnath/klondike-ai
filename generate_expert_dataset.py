import argparse
import json
import os
import random
from typing import Dict, List, Tuple, Set
from tqdm import trange
import numpy as np

from env.klondike_env import KlondikeEnv
from env.reward import compute_reward


SUITS = ["H", "D", "C", "S"]


def _normalize_card(card: str) -> Tuple[str, str]:
    """Return (rank, suit) tuple with suit uppercased."""
    card = str(card)
    rank, suit = card[:-1], card[-1]
    return rank, suit.upper()


def get_hidden_cards(state: Dict) -> Set[Tuple[str, str]]:
    """Extract the set of hidden cards from a state."""
    hidden: Set[Tuple[str, str]] = set()
    tableau = state.get("tableau", [])
    for col in tableau:
        if isinstance(col, dict):
            cards = col.get("cards", [])
            down = int(col.get("face_down", 0))
            for c in cards[:down]:
                if isinstance(c, str):
                    hidden.add(_normalize_card(c))
        elif isinstance(col, list):
            for c in col:
                if isinstance(c, str) and c[-1].islower():
                    hidden.add(_normalize_card(c))
    return hidden


def count_empty_columns(state: Dict) -> int:
    """Count tableau columns that are completely empty."""
    empty = 0
    tableau = state.get("tableau", [])
    for col in tableau:
        if isinstance(col, dict):
            cards = col.get("cards", [])
            if not cards and int(col.get("face_down", 0)) == 0:
                empty += 1
        elif isinstance(col, list):
            if len(col) == 0:
                empty += 1
    return empty


def extract_foundations(state: Dict) -> Dict[str, int]:
    """Return foundation counts per suit."""
    res: Dict[str, int] = {}
    foundations = state.get("foundations", [])
    for i, stack in enumerate(foundations):
        suit = SUITS[i] if i < len(SUITS) else str(i)
        if isinstance(stack, list):
            res[suit] = len(stack)
        else:
            try:
                res[suit] = int(stack)
            except Exception:
                res[suit] = 0
    return res

try:
    from klondike_core import solve_klondike, move_index, shuffle_seed
except Exception:  # pragma: no cover - fallback when extension missing
    from klondike_core import solve_klondike, move_index
    shuffle_seed = None


def generate_games(num_games: int, output: str) -> None:
    """Generate expert dataset using the optimal solver."""
    os.makedirs(os.path.dirname(output), exist_ok=True)
    transitions: List[Dict] = []
    for _ in trange(num_games, desc="games"):
        seed = str(shuffle_seed()) if shuffle_seed else str(random.randint(0, 2**32 - 1))
        solution = json.loads(solve_klondike(seed))
        moves: List[str] = solution.get("moves", [])
        env = KlondikeEnv(seed=seed)
        env.reset(seed)
        prev_state = env.state
        for mv_json in moves:
            action = move_index(mv_json)
            _, rew, done, _ = env.step(action)

            prev_state_dict = json.loads(prev_state)
            next_state_dict = json.loads(env.state)

            tags = {
                "reveals_hidden_card": len(get_hidden_cards(next_state_dict)) < len(get_hidden_cards(prev_state_dict)),
                "frees_column": count_empty_columns(next_state_dict) > count_empty_columns(prev_state_dict),
                "foundation_progress": sum(extract_foundations(next_state_dict).values()) > sum(extract_foundations(prev_state_dict).values()),
                "is_critical": compute_reward(prev_state, action, env.state, done) > 0,
            }

            transitions.append(
                {
                    "state": prev_state_dict,
                    "action": {
                        "index": int(action),
                        "semantic_tags": tags,
                    },
                    "next_state": next_state_dict,
                    "reward": float(rew),
                    "done": bool(done),
                }
            )

            prev_state = env.state
            if done:
                break

    np.savez_compressed(output, transitions=np.array(transitions, dtype=object))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate expert dataset")
    parser.add_argument("--games", type=int, default=100, help="Number of games")
    parser.add_argument(
        "--output",
        type=str,
        default="data/expert_dataset.npz",
        help="Output path",
    )
    args = parser.parse_args()
    generate_games(args.games, args.output)


if __name__ == "__main__":
    main()
