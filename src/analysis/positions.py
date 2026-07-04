"""
positions.py

Shared helpers for parsing player position strings and classifying pitchers.

Position data arrives in several shapes across the codebase: a bare string
("SP"), a composite string ("SP/RP", "1B/OF"), a comma-separated string
("SP,P"), or a list/tuple/ndarray of position strings. These helpers normalize
all of those into a flat list of upper-cased tokens so pitcher detection is
consistent everywhere.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Position tokens that indicate a pitcher. Includes the bare "P" slot.
_PITCHER_POSITIONS: frozenset[str] = frozenset({"SP", "RP", "P"})


def parse_positions(positions: Any) -> list[str]:
    """Normalize a position value into a flat list of upper-cased tokens.

    Handles ``None``, bare strings, composite/comma-separated strings
    ("SP/RP", "1B,OF"), lists, tuples, sets, and numpy arrays (including
    0-d scalar arrays).
    """
    if positions is None:
        return []
    if isinstance(positions, np.ndarray):
        if positions.ndim == 0:
            return parse_positions(positions.item())
        tokens: list[str] = []
        for p in positions:
            tokens.extend(parse_positions(p))
        return tokens
    if isinstance(positions, (list, tuple, set, frozenset)):
        tokens = []
        for p in positions:
            tokens.extend(parse_positions(p))
        return tokens
    if isinstance(positions, str):
        parts = positions.replace("/", ",").split(",")
        return [p.strip().upper() for p in parts if p.strip()]
    text = str(positions).strip()
    return [text.upper()] if text else []


def is_pitcher(positions: Any) -> bool:
    """Return True when any parsed position token is a pitcher slot (P/SP/RP)."""
    return bool(set(parse_positions(positions)) & _PITCHER_POSITIONS)
