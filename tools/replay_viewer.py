import argparse
import json
import random
import time
from typing import List, Tuple, Optional

import numpy as np

try:
    from klondike_core import new_game, play_move, move_from_index
except Exception as exc:  # pragma: no cover - handle missing build
    raise ImportError(
        "klondike_core module is required. Build the Rust extension first"
    ) from exc


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load NPZ expert dataset."""
    data = np.load(path, allow_pickle=True)
    actions = data["actions"]
    dones = data["dones"]
    intentions = data["intentions"] if "intentions" in data else None
    seeds = data["seeds"] if "seeds" in data else None
    return actions, dones, intentions, seeds


def split_episodes(dones: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of (start, end) indices for episodes."""
    ends = np.where(dones)[0]
    starts = np.concatenate(([0], ends[:-1] + 1))
    return list(zip(starts, ends + 1))


def format_state(state: str) -> str:
    """Return a simple textual representation of game state."""
    data = json.loads(state)
    lines = []
    founds = [" ".join(stack) if stack else "[]" for stack in data.get("foundations", [])]
    lines.append("Foundations: " + " | ".join(founds))
    waste = data.get("waste", [])
    lines.append(f"Waste(top3): {' '.join(waste[-3:])}")
    lines.append(f"Stock left: {len(data.get('stock', []))}")
    tableau = data.get("tableau", [])
    max_len = max((len(col) for col in tableau), default=0)
    for i in range(max_len):
        row = []
        for col in tableau:
            row.append(col[i] if i < len(col) else "  ")
        lines.append(" ".join(f"{c:>3}" for c in row))
    return "\n".join(lines)


def replay_episode(actions: np.ndarray, intentions: Optional[np.ndarray], seed: Optional[str], delay: float) -> None:
    """Replay one episode given actions and optional intentions."""
    state = new_game(seed) if seed is not None else new_game()
    for idx, action in enumerate(actions, 1):
        move = move_from_index(int(action))
        if move is None:
            print(f"Step {idx}: invalid action index {action}")
            break
        print("=" * 40)
        print(f"Step {idx}")
        print(format_state(state))
        intent = intentions[idx - 1] if intentions is not None else ""
        if intent:
            print(f"Action: {move} | Intention: {intent}")
        else:
            print(f"Action: {move}")
        state, valid = play_move(state, move)
        if not valid:
            print("Move was invalid according to the engine. Stopping replay.")
            break
        if delay <= 0:
            input("Press Enter to continue...")
        else:
            time.sleep(delay)
    print("=" * 40)
    print("Final state:")
    print(format_state(state))


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay expert dataset episodes")
    parser.add_argument("--dataset", required=True, help="Path to .npz dataset")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--game", type=int, help="Index of the game to replay (0-based)")
    group.add_argument("--random", action="store_true", help="Pick a random game")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between moves (0 for manual)")
    args = parser.parse_args()

    actions, dones, intentions, seeds = load_dataset(args.dataset)
    episodes = split_episodes(dones)
    if not episodes:
        raise RuntimeError("No episodes found in dataset")

    if args.random:
        idx = random.randrange(len(episodes))
    else:
        idx = args.game if args.game is not None else 0
        idx = max(0, min(idx, len(episodes) - 1))
    start, end = episodes[idx]
    ep_actions = actions[start:end]
    ep_intents = intentions[start:end] if intentions is not None else None
    seed = None
    if seeds is not None and idx < len(seeds):
        seed_val = seeds[idx]
        seed = str(seed_val) if seed_val is not None else None

    print(f"Replaying game {idx} with {len(ep_actions)} moves")
    replay_episode(ep_actions, ep_intents, seed, args.delay)


if __name__ == "__main__":
    main()
