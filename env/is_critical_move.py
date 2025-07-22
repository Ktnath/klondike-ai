import json
import logging
from typing import List, Dict


def _parse_state(state: str) -> Dict:
    """Parse a JSON game state string."""
    try:
        data = json.loads(state)
    except Exception as exc:  # pragma: no cover - invalid json
        raise ValueError("Invalid JSON state") from exc
    if not isinstance(data, dict):
        raise ValueError("State should be a JSON object")
    if "foundations" not in data or "tableau" not in data:
        logging.warning("State missing 'foundations' or 'tableau' fields")
    return data


def _foundation_count(state: Dict) -> int:
    return sum(len(stack) for stack in state.get("foundations", []))


def _foundations_started(state: Dict) -> int:
    return sum(1 for stack in state.get("foundations", []) if len(stack) > 0)


def _turned_over(before: Dict, after: Dict) -> int:
    turned = 0
    b_tab = before.get("tableau", [])
    a_tab = after.get("tableau", [])
    for i in range(min(len(b_tab), len(a_tab))):
        b_down = b_tab[i].get("face_down", 0)
        a_down = a_tab[i].get("face_down", 0)
        if a_down < b_down:
            turned += b_down - a_down
    return turned


def _rank(card: str) -> str:
    return card[:-1]


def _king_to_empty(before: Dict, after: Dict) -> int:
    count = 0
    b_tab = before.get("tableau", [])
    a_tab = after.get("tableau", [])
    for i in range(min(len(b_tab), len(a_tab))):
        b_col = b_tab[i]
        a_col = a_tab[i]
        if (
            b_col.get("face_down", 0) == 0
            and len(b_col.get("cards", [])) == 0
            and len(a_col.get("cards", [])) > 0
        ):
            first = a_col.get("cards", [])[0]
            if first and _rank(str(first)) == "K":
                count += 1
    return count


def _emptied_columns(before: Dict, after: Dict) -> int:
    count = 0
    b_tab = before.get("tableau", [])
    a_tab = after.get("tableau", [])
    for i in range(min(len(b_tab), len(a_tab))):
        b_col = b_tab[i]
        a_col = a_tab[i]
        if len(b_col.get("cards", [])) > 0 and len(a_col.get("cards", [])) == 0 and a_col.get("face_down", 0) == 0:
            count += 1
    return count


def is_critical_move(state_before: str, state_after: str) -> float:
    """Heuristic criticality score between two JSON encoded states."""
    before = _parse_state(state_before)
    after = _parse_state(state_after)

    score = 0.0

    # 1. Card moved to foundation
    diff_foundations = _foundation_count(after) - _foundation_count(before)
    if diff_foundations > 0:
        piles_started = _foundations_started(after)
        score += diff_foundations * 0.5 * piles_started

    # 2. Card flipped in tableau
    turned = _turned_over(before, after)
    score += turned * 1.0

    # 3. King moved to empty column
    kings = _king_to_empty(before, after)
    score += kings * 0.7

    # 4. Column emptied
    emptied = _emptied_columns(before, after)
    score += emptied * 0.5

    # 5. Game won
    if after.get("is_won") or _foundation_count(after) == 52:
        score += 10.0

    return float(score)

