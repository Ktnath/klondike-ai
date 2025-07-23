import argparse
import json
import os
from typing import Dict, List

from intention_utils import simplify_intention, filter_ambiguous

import numpy as np
from tqdm import trange

try:
    import torch
except Exception:  # pragma: no cover - optional
    torch = None

from klondike_core import (
    new_game,
    legal_moves,
    play_move,
    move_index,
    compute_base_reward_json,
    encode_observation,
    foundation_count,
    is_won,
)

try:
    from klondike_ai import run_mcts_for_state
except Exception:  # pragma: no cover - optional
    run_mcts_for_state = None

from env.state_utils import count_empty_columns
from train.train_dqn import DQN, DuelingDQN

INTENT_REVEAL = "Révéler une carte cachée"
INTENT_FOUNDATION = "Monter à la fondation"
INTENT_KING_EMPTY = "Déplacer un roi sur colonne vide"
INTENT_STACK = "Ranger carte sur une autre"

def _infer_intention(before: str, mv: str, after: str) -> str:
    """Infer the intention of a move using JSON states."""
    before_dict = json.loads(before)
    after_dict = json.loads(after)

    if mv.startswith("R"):
        return INTENT_REVEAL

    try:
        if foundation_count(after) > foundation_count(before):
            return INTENT_FOUNDATION
    except Exception:
        pass

    parts = mv.split()
    mv_type = parts[0] if parts else ""
    idx = int(parts[1]) if len(parts) > 1 else 0

    if mv_type in {"DP", "SP"}:
        before_empty = count_empty_columns(before_dict)
        after_empty = count_empty_columns(after_dict)
        if idx // 4 == 12 and after_empty < before_empty:
            return INTENT_KING_EMPTY
        return INTENT_STACK

    if mv_type in {"DS", "PS"}:
        return INTENT_FOUNDATION

    return INTENT_STACK


def _select_action_dqn(model, obs: np.ndarray, valid: Dict[int, str]) -> int:
    with torch.no_grad():
        q = model(torch.tensor(obs, dtype=torch.float32))
    actions = list(valid.keys())
    q_valid = q[actions]
    best = int(actions[int(torch.argmax(q_valid).item())])
    return best


def _select_action_mcts(state_json: str, valid: Dict[int, str]) -> int:
    if run_mcts_for_state is None:
        raise RuntimeError("klondike_ai not available for MCTS policy")
    try:
        res = run_mcts_for_state(state_json, 50)
        action = int(json.loads(res))
    except Exception:
        action = None
    if action not in valid:
        action = next(iter(valid))
    return action


def generate_self_play(model_path: str | None, output: str, episodes: int, use_mcts: bool) -> None:
    os.makedirs(os.path.dirname(output), exist_ok=True)

    observations: List[List[float]] = []
    actions: List[int] = []
    rewards: List[float] = []
    dones: List[bool] = []
    intentions: List[str] = []

    model = None
    if model_path and not use_mcts:
        if torch is None:
            raise RuntimeError("PyTorch is required to load a DQN model")
        dummy_state = new_game()
        obs_dim = len(encode_observation(dummy_state))
        action_dim = 96
        try:
            model = DuelingDQN(obs_dim, action_dim)
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        except Exception:
            model = DQN(obs_dim, action_dim)
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

    for _ in trange(episodes, desc="episodes"):
        state = new_game()
        done = False
        while not done:
            encoded = json.loads(state)["encoded"]
            moves = legal_moves(encoded)
            if not moves:
                break
            mapping = {move_index(mv): mv for mv in moves}
            obs = encode_observation(state)
            if use_mcts:
                action = _select_action_mcts(state, mapping)
            else:
                action = _select_action_dqn(model, np.array(obs, dtype=np.float32), mapping)
            mv = mapping.get(action, moves[0])
            next_state, _ = play_move(state, mv)
            reward = compute_base_reward_json(next_state)
            done = bool(is_won(json.loads(next_state)["encoded"]))
            intention = _infer_intention(state, mv, next_state)

            observations.append(list(obs))
            actions.append(int(action))
            rewards.append(float(reward))
            dones.append(bool(done))
            intentions.append(simplify_intention(str(intention)))

            state = next_state
            if done:
                break

    filtered = filter_ambiguous(intentions)
    mask = [i is not None for i in filtered]
    observations = [o for o, m in zip(observations, mask) if m]
    actions = [a for a, m in zip(actions, mask) if m]
    rewards = [r for r, m in zip(rewards, mask) if m]
    dones = [d for d, m in zip(dones, mask) if m]
    intentions = [i for i in filtered if i is not None]

    np.savez_compressed(
        output,
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.int64),
        rewards=np.array(rewards, dtype=np.float32),
        dones=np.array(dones, dtype=bool),
        intentions=np.array(intentions, dtype=object),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate self-play dataset")
    parser.add_argument("--model_path", type=str, default=None, help="Path to DQN model (.pt)")
    parser.add_argument("--output", type=str, default="data/self_play.npz", help="Output file")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--use_mcts", action="store_true", help="Use MCTS policy instead of DQN")
    args = parser.parse_args()

    generate_self_play(args.model_path, args.output, args.episodes, args.use_mcts)


if __name__ == "__main__":
    main()
