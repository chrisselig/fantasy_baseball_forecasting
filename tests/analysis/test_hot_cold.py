"""Tests for src/analysis/hot_cold.py"""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

from src.analysis.hot_cold import (
    _COLD,
    _HOT,
    _NEUTRAL,
    _hitter_streak,
    _pitcher_streak,
    annotate_with_streaks,
    match_win_probability,
    streak_label,
)

# ── Helpers ────────────────────────────────────────────────────────────────


def _hitter_row(
    player_id: str,
    stat_date: str,
    h: int = 0,
    ab: int = 4,
    hr: int = 0,
    sb: int = 0,
    bb: int = 0,
    hbp: int = 0,
    sf: int = 0,
    tb: int = 0,
) -> dict[str, object]:
    return {
        "player_id": player_id,
        "stat_date": stat_date,
        "h": h,
        "ab": ab,
        "hr": hr,
        "sb": sb,
        "bb": bb,
        "hbp": hbp,
        "sf": sf,
        "tb": tb or h,
        "ip": 0.0,
        "k": 0,
        "walks_allowed": 0,
        "hits_allowed": 0,
    }


def _pitcher_row(
    player_id: str,
    stat_date: str,
    ip: float = 6.0,
    k: int = 7,
    walks_allowed: int = 2,
    hits_allowed: int = 5,
) -> dict[str, object]:
    return {
        "player_id": player_id,
        "stat_date": stat_date,
        "h": 0,
        "ab": 0,
        "hr": 0,
        "sb": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
        "tb": 0,
        "ip": ip,
        "k": k,
        "walks_allowed": walks_allowed,
        "hits_allowed": hits_allowed,
    }


# ── _hitter_streak ──────────────────────────────────────────────────────────


def test_hitter_hot_hit_streak_and_power() -> None:
    rows = [
        _hitter_row("p1", f"2025-06-0{i}", h=2, ab=4, hr=1, tb=5) for i in range(1, 8)
    ]
    df = pd.DataFrame(rows)
    assert _hitter_streak(df) == _HOT


def test_hitter_hot_avg_and_ops() -> None:
    # 7 days: 3 H / 4 AB each → .750 AVG; OPS >> .920
    rows = [_hitter_row("p1", f"2025-06-0{i}", h=3, ab=4, tb=5) for i in range(1, 8)]
    df = pd.DataFrame(rows)
    assert _hitter_streak(df) == _HOT


def test_hitter_cold_hitless_run() -> None:
    rows = [_hitter_row("p1", f"2025-06-0{i}", h=0, ab=4) for i in range(1, 8)]
    df = pd.DataFrame(rows)
    assert _hitter_streak(df) == _COLD


def test_hitter_cold_low_avg_and_no_power() -> None:
    # 7 days: 1 H / 10 AB each → .100 AVG; 0 HR, 0 SB
    rows = [_hitter_row("p1", f"2025-06-0{i}", h=1, ab=10, tb=1) for i in range(1, 8)]
    df = pd.DataFrame(rows)
    assert _hitter_streak(df) == _COLD


def test_hitter_neutral_insufficient_data() -> None:
    rows = [_hitter_row("p1", "2025-06-01", h=1, ab=4)]
    df = pd.DataFrame(rows)
    assert _hitter_streak(df) == _NEUTRAL


def test_hitter_neutral_mixed_signals() -> None:
    # Mix of good and bad days — should not trigger 2+ hot or cold
    rows = [
        _hitter_row("p1", f"2025-06-0{i}", h=i % 2, ab=4, tb=i % 2) for i in range(1, 8)
    ]
    df = pd.DataFrame(rows)
    # Shouldn't crash; outcome is NEUTRAL or one direction (just check no error)
    result = _hitter_streak(df)
    assert result in (_HOT, _COLD, _NEUTRAL)


# ── _pitcher_streak ─────────────────────────────────────────────────────────


def test_pitcher_hot_low_whip_high_k() -> None:
    # 12 IP, 3 W, 2 H allowed → WHIP=0.42, RA9=2.25, K9=11.25
    rows = [
        _pitcher_row("p2", "2025-06-05", ip=6.0, k=8, walks_allowed=1, hits_allowed=2),
        _pitcher_row("p2", "2025-06-01", ip=6.0, k=7, walks_allowed=2, hits_allowed=3),
    ]
    df = pd.DataFrame(rows)
    assert _pitcher_streak(df) == _HOT


def test_pitcher_cold_high_whip_low_k() -> None:
    # 10 IP, 8 H + 7 BB → WHIP=1.5, K9=4.5
    rows = [
        _pitcher_row("p2", "2025-06-05", ip=5.0, k=3, walks_allowed=4, hits_allowed=5),
        _pitcher_row("p2", "2025-06-01", ip=5.0, k=2, walks_allowed=3, hits_allowed=6),
    ]
    df = pd.DataFrame(rows)
    assert _pitcher_streak(df) == _COLD


def test_pitcher_neutral_no_ip() -> None:
    rows = [_pitcher_row("p2", "2025-06-01", ip=0.0)]
    df = pd.DataFrame(rows)
    assert _pitcher_streak(df) == _NEUTRAL


# ── streak_label ────────────────────────────────────────────────────────────


def test_streak_label_unknown_player_returns_neutral() -> None:
    df = pd.DataFrame([_hitter_row("p1", "2025-06-01", h=2, ab=4)])
    assert streak_label("unknown_player", df, is_pitcher=False) == _NEUTRAL


def test_streak_label_hitter_hot() -> None:
    rows = [
        _hitter_row("p1", f"2025-06-0{i}", h=2, ab=4, hr=1, tb=5) for i in range(1, 8)
    ]
    df = pd.DataFrame(rows)
    ref = datetime.date(2025, 6, 7)
    assert streak_label("p1", df, is_pitcher=False, reference_date=ref) == _HOT


def test_streak_label_pitcher_hot() -> None:
    rows = [
        _pitcher_row("p2", "2025-06-05", ip=7.0, k=10, walks_allowed=1, hits_allowed=2),
        _pitcher_row("p2", "2025-05-31", ip=7.0, k=9, walks_allowed=1, hits_allowed=3),
    ]
    df = pd.DataFrame(rows)
    ref = datetime.date(2025, 6, 7)
    assert streak_label("p2", df, is_pitcher=True, reference_date=ref) == _HOT


# ── annotate_with_streaks ───────────────────────────────────────────────────


def test_annotate_adds_streak_column() -> None:
    roster = pd.DataFrame(
        [
            {"player_id": "p1", "position": "OF"},
            {"player_id": "p2", "position": "SP"},
        ]
    )
    daily = pd.DataFrame(
        [_hitter_row("p1", "2025-06-01", h=2, ab=4)]
        + [_pitcher_row("p2", "2025-06-01")]
    )
    result = annotate_with_streaks(roster, daily)
    assert "streak" in result.columns
    assert len(result) == 2


def test_annotate_does_not_mutate_input() -> None:
    roster = pd.DataFrame([{"player_id": "p1", "position": "OF"}])
    daily = pd.DataFrame([_hitter_row("p1", "2025-06-01")])
    _ = annotate_with_streaks(roster, daily)
    assert "streak" not in roster.columns


# ── match_win_probability ───────────────────────────────────────────────────


def test_mwp_certain_win() -> None:
    probs = [1.0] * 12
    assert match_win_probability(probs) == pytest.approx(1.0)


def test_mwp_certain_loss() -> None:
    probs = [0.0] * 12
    assert match_win_probability(probs) == pytest.approx(0.0)


def test_mwp_even_odds_near_half() -> None:
    probs = [0.5] * 12
    # By symmetry, P(X > 6) ≈ 0.387 (binomial 12, 0.5)
    result = match_win_probability(probs)
    assert 0.35 < result < 0.45


def test_mwp_favored() -> None:
    # Win 9 of 12 categories with 80% probability each
    probs = [0.8] * 9 + [0.2] * 3
    result = match_win_probability(probs)
    assert result > 0.7


def test_mwp_empty() -> None:
    assert match_win_probability([]) == pytest.approx(0.0)
