import json

from klondike_core import (
    new_game,
    play_move,
    legal_moves,
    compute_base_reward_json,
    encode_state_to_json,
    move_index,
    move_from_index,
    shuffle_seed,
)


def banner(title: str) -> None:
    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)


def main() -> None:
    # new_game
    try:
        state = new_game()
        banner("new_game")
        print(state)
    except Exception as e:
        state = None
        print(f"new_game() failed: {e}")

    # legal_moves
    try:
        encoded = json.loads(state)["encoded"] if state else "0"
        moves = legal_moves(encoded)
        banner("legal_moves")
        print(moves)
    except Exception as e:
        moves = []
        print(f"legal_moves() failed: {e}")

    # play_move
    try:
        if state and moves:
            next_state, valid = play_move(state, moves[0])
            banner("play_move")
            print(next_state)
            print("valid:", valid)
            state = next_state
        else:
            banner("play_move")
            print("No moves available")
    except Exception as e:
        print(f"play_move() failed: {e}")

    # compute_base_reward_json
    try:
        reward = compute_base_reward_json(state) if state else None
        banner("compute_base_reward_json")
        print(reward)
    except Exception as e:
        print(f"compute_base_reward_json() failed: {e}")

    # encode_state_to_json
    try:
        if state:
            encoded = json.loads(state)["encoded"]
        result = encode_state_to_json(encoded)
        banner("encode_state_to_json")
        print(result)
    except Exception as e:
        print(f"encode_state_to_json() failed: {e}")

    # move_index
    try:
        sample_move = moves[0] if moves else "DS 0"
        idx = move_index(sample_move)
        banner("move_index")
        print(idx)
    except Exception as e:
        idx = 0
        print(f"move_index() failed: {e}")

    # move_from_index
    try:
        mv = move_from_index(idx)
        banner("move_from_index")
        print(mv)
    except Exception as e:
        print(f"move_from_index() failed: {e}")

    # shuffle_seed
    try:
        seed = shuffle_seed()
        banner("shuffle_seed")
        print(seed)
    except Exception as e:
        print(f"shuffle_seed() failed: {e}")


if __name__ == "__main__":
    main()
