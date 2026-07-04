"""
tests/analysis/test_positions.py

Tests for the shared position-parsing helper.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.positions import is_pitcher, parse_positions


@pytest.mark.parametrize(
    "value,expected",
    [
        ("P", True),
        ("SP", True),
        ("RP", True),
        ("SP/RP", True),
        ("sp/rp", True),
        ("SP,P", True),
        ("1B/OF", False),
        ("C", False),
        ("2B/SS", False),
        (["SP", "RP"], True),
        (["1B", "OF"], False),
        (["1B/OF", "P"], True),
        (("C",), False),
        (np.array(["SP", "OF"]), True),
        (np.array("P"), True),
        (np.array("1B"), False),
        (None, False),
        ("", False),
    ],
)
def test_is_pitcher(value: object, expected: bool) -> None:
    assert is_pitcher(value) is expected


def test_bare_p_is_detected() -> None:
    # Regression: the old hot_cold helper missed a bare "P" slot.
    assert is_pitcher("P") is True


def test_parse_positions_composite_and_comma() -> None:
    assert parse_positions("SP/RP") == ["SP", "RP"]
    assert parse_positions("1B, OF") == ["1B", "OF"]
    assert parse_positions(None) == []
    assert parse_positions(["1B/OF", "P"]) == ["1B", "OF", "P"]
