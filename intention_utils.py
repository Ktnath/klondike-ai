from __future__ import annotations

"""Utilities for intention processing."""

from collections import Counter
from typing import List, Optional, Iterable, Union
import argparse
from pathlib import Path


# This threshold is intentionally small so that common intentions are kept
_MIN_FREQ = 2

# Mapping of specific solver intention labels to strategic groups
INTENTION_HIERARCHY = {
    "REVEAL": "REVEAL",
    "MOVE_TO_FOUNDATION": "FOUNDATION",
    "MOVE_KING_TO_EMPTY": "KING_TO_EMPTY",
    "MOVE_TO_TABLEAU": "BUILD_STACK",
    "MOVE_FROM_DECK": "BUILD_STACK",
    "MOVE_TO_EMPTY_COLUMN": "KING_TO_EMPTY",
    "CYCLE_DECK": "CYCLE",
    "UNKNOWN": "OTHER",
}


def simplify_intention(raw: str) -> str:
    """Return a canonical version of an intention string."""
    text = str(raw).strip().lower()
    if not text:
        return ""
    if "reveler" in text or "révéler" in text:
        return "Révéler"
    if "fondation" in text or "monter" in text:
        return "Monter en fondation"
    if "roi" in text and "vide" in text:
        return "Déplacer vers pile vide"
    if "deplacer" in text or "déplacer" in text or "ranger" in text or "pile" in text:
        return "Déplacer vers pile"
    return raw.strip()


def filter_ambiguous(intentions: List[str], min_freq: int = _MIN_FREQ) -> List[Optional[str]]:
    """Replace rare or empty intentions with ``None``."""
    counts = Counter(intentions)
    return [i if i and counts[i] >= min_freq else None for i in intentions]


def group_into_hierarchy(intentions: Union[str, Iterable[str]]) -> Union[str, List[str]]:
    """Return hierarchical categories for each intention.

    Accepts a single string or any iterable of strings, including generators.
    """

    category_map = {
        "Révéler": "Information",
        "Monter en fondation": "Score",
        "Déplacer vers pile": "Réorganisation",
        "Déplacer vers pile vide": "Réorganisation",
    }

    def _map_one(value: str) -> str:
        if value.upper() in INTENTION_HIERARCHY:
            return INTENTION_HIERARCHY.get(value.upper(), "OTHER")
        base = simplify_intention(value)
        cat = category_map.get(base, "Autre")
        return f"{base} \u2192 {cat}"

    if isinstance(intentions, str):
        return _map_one(intentions)
    if not isinstance(intentions, Iterable):
        raise TypeError("Input must be a string or iterable of strings.")
    return [_map_one(i) for i in intentions]


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Group fine-grained intentions into macro categories"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-f",
        "--file",
        type=Path,
        help="Path to text file containing intentions (comma-separated or one per line).",
    )
    group.add_argument(
        "-i",
        "--input",
        type=str,
        help="Comma-separated list of intentions provided directly on the command line.",
    )
    return parser.parse_args()


def _read_intentions(args: argparse.Namespace) -> List[str]:
    """Load intentions from a file or from the inline ``--input`` string."""
    if args.file:
        text = args.file.read_text(encoding="utf-8").strip()
    else:
        text = args.input.strip()

    # Determine whether the text contains commas or new lines
    if "," in text and not "\n" in text:
        raw = [t.strip() for t in text.split(",") if t.strip()]
    else:
        lines = text.splitlines()
        raw = []
        for line in lines:
            if "," in line:
                raw.extend(t.strip() for t in line.split(",") if t.strip())
            elif line.strip():
                raw.append(line.strip())
    return raw


if __name__ == "__main__":
    arguments = _parse_args()
    intentions = _read_intentions(arguments)
    grouped = group_into_hierarchy(intentions)
    for g in grouped:
        print(g)
