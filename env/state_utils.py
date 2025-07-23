"""Utility helpers for analyzing JSON game states."""
from __future__ import annotations

import json
from typing import Dict, List, Tuple, Set

SUITS = ["H", "D", "C", "S"]


def _normalize_card(card: str) -> Tuple[str, str]:
    """Return (rank, suit) tuple with suit uppercased."""
    card = str(card)
    rank, suit = card[:-1], card[-1]
    return rank, suit.upper()


def get_hidden_cards(state: Dict) -> Set[Tuple[str, str]]:
    """Return the set of hidden cards in the tableau."""
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
    """Return the number of cards in each foundation."""
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
