"""Simple demo script for state analysis utilities."""
from __future__ import annotations

import json

from env.klondike_env import KlondikeEnv
from env import get_hidden_cards, count_empty_columns, extract_foundations


def main() -> None:
    env = KlondikeEnv()
    env.reset()
    state_dict = json.loads(env.state)

    hidden = sorted(list(get_hidden_cards(state_dict)))
    empty = count_empty_columns(state_dict)
    foundations = extract_foundations(state_dict)

    print(f"Hidden cards: {hidden}")
    print(f"Empty columns: {empty}")
    print(f"Foundations: {foundations}")


if __name__ == "__main__":
    main()
