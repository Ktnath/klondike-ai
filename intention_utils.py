from __future__ import annotations

"""Utilities for intention processing."""

from collections import Counter
from typing import List, Optional


# This threshold is intentionally small so that common intentions are kept
_MIN_FREQ = 2


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


def group_into_hierarchy(intentions: List[str]) -> List[str]:
    """Return hierarchical categories for each intention."""
    category_map = {
        "Révéler": "Information",
        "Monter en fondation": "Score",
        "Déplacer vers pile": "Réorganisation",
        "Déplacer vers pile vide": "Réorganisation",
    }
    result = []
    for i in intentions:
        if not i:
            result.append(i)
            continue
        base = simplify_intention(i)
        cat = category_map.get(base, "Autre")
        result.append(f"{base} \u2192 {cat}")
    return result
