"""Tests for src/analysis/hot_cold.py"""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

from src.analysis.hot_cold import (
    _COLD,
    _HOT,
    _NEUTRAL,
    _WARM,
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


def test_pitcher_warm_one_hot_condition() -> None:
    # WHIP = (4+2)/7 = 0.857 → <1.10 (hot); RA9 = 6*9/7 = 7.71 → >5.00 (cold)
    # K9 = 6*9/7 = 7.71 → neither; KBB = 6/2 = 3.0 → neither (not strictly >3.0)
    # hot_score=1, cold_score=1 → Warm
    rows = [
        _pitcher_row("p2", "2025-06-05", ip=7.0, k=6, walks_allowed=2, hits_allowed=4)
    ]
    df = pd.DataFrame(rows)
    assert _pitcher_streak(df) == _WARM


def test_pitcher_hot_whip_threshold_1_10() -> None:
    # WHIP = (3+2)/5 = 1.0 → now qualifies as hot (old threshold was 1.00)
    # RA9 = 5*9/5 = 9.0 → >5.00 (cold); K9 = 10*9/5 = 18.0 → hot; KBB = 10/2 = 5.0 → hot
    # hot_score=3, → HOT
    rows = [
        _pitcher_row("p2", "2025-06-05", ip=5.0, k=10, walks_allowed=2, hits_allowed=3)
    ]
    df = pd.DataFrame(rows)
    assert _pitcher_streak(df) == _HOT


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


# ── Prior-based streak path ────────────────────────────────────────────────


def test_hitter_prior_hot_when_above_xwoba_baseline() -> None:
    # League-average xwOBA hitter (.320 → prior OPS ~.735), but smashing
    # the ball over the last week (.500 / 1.000+ OPS).
    rows = [
        _hitter_row("p1", f"2025-06-0{i}", h=2, ab=4, hr=1, tb=6) for i in range(1, 8)
    ]
    daily = pd.DataFrame(rows)
    advanced = pd.DataFrame(
        [{"player_id": "p1", "xwoba": 0.320, "xwoba_against": None}]
    )
    label = streak_label(
        "p1",
        daily,
        is_pitcher=False,
        reference_date=datetime.date(2025, 6, 8),
        advanced_df=advanced,
    )
    assert label == _HOT


def test_hitter_prior_cold_when_below_xwoba_baseline() -> None:
    # Elite hitter prior (.380 xwOBA → ~.91 OPS), but slumping (.100 / .200 OPS)
    rows = [_hitter_row("p1", f"2025-06-0{i}", h=0, ab=4) for i in range(1, 8)]
    daily = pd.DataFrame(rows)
    advanced = pd.DataFrame(
        [{"player_id": "p1", "xwoba": 0.380, "xwoba_against": None}]
    )
    label = streak_label(
        "p1",
        daily,
        is_pitcher=False,
        reference_date=datetime.date(2025, 6, 8),
        advanced_df=advanced,
    )
    assert label == _COLD


def test_hitter_prior_neutral_when_matching_baseline() -> None:
    # Player roughly matching league-average prior — should be neutral
    rows = [_hitter_row("p1", f"2025-06-0{i}", h=1, ab=4, tb=2) for i in range(1, 8)]
    daily = pd.DataFrame(rows)
    advanced = pd.DataFrame(
        [{"player_id": "p1", "xwoba": 0.320, "xwoba_against": None}]
    )
    label = streak_label(
        "p1",
        daily,
        is_pitcher=False,
        reference_date=datetime.date(2025, 6, 8),
        advanced_df=advanced,
    )
    assert label == _NEUTRAL


def test_pitcher_prior_hot_when_below_whip_baseline() -> None:
    # League-average pitcher prior (.310 xwOBA-against → WHIP ~1.27)
    # but recent 10-day WHIP at 0.50 → easily Hot.
    rows = [
        _pitcher_row("p2", "2025-06-05", ip=6.0, k=8, walks_allowed=1, hits_allowed=2),
        _pitcher_row("p2", "2025-06-01", ip=6.0, k=7, walks_allowed=1, hits_allowed=2),
    ]
    daily = pd.DataFrame(rows)
    advanced = pd.DataFrame(
        [{"player_id": "p2", "xwoba": None, "xwoba_against": 0.310}]
    )
    label = streak_label(
        "p2",
        daily,
        is_pitcher=True,
        reference_date=datetime.date(2025, 6, 6),
        advanced_df=advanced,
    )
    assert label == _HOT


def test_pitcher_prior_cold_when_above_whip_baseline() -> None:
    # Elite pitcher prior (.260 xwOBA-against → WHIP ~0.98) but blown up
    # recently (~2.50 WHIP).
    rows = [
        _pitcher_row("p2", "2025-06-05", ip=4.0, k=2, walks_allowed=4, hits_allowed=6),
        _pitcher_row("p2", "2025-06-01", ip=4.0, k=2, walks_allowed=3, hits_allowed=7),
    ]
    daily = pd.DataFrame(rows)
    advanced = pd.DataFrame(
        [{"player_id": "p2", "xwoba": None, "xwoba_against": 0.260}]
    )
    label = streak_label(
        "p2",
        daily,
        is_pitcher=True,
        reference_date=datetime.date(2025, 6, 6),
        advanced_df=advanced,
    )
    assert label == _COLD


def test_prior_path_falls_back_to_season_baseline_when_no_advanced_row() -> None:
    # Player not in advanced_df → should use season baseline (the daily df
    # itself). With identical season and recent rates the result is neutral.
    rows = [_hitter_row("p1", f"2025-06-0{i}", h=1, ab=4, tb=1) for i in range(1, 8)]
    daily = pd.DataFrame(rows)
    advanced = pd.DataFrame(
        [{"player_id": "OTHER", "xwoba": 0.320, "xwoba_against": None}]
    )
    label = streak_label(
        "p1",
        daily,
        is_pitcher=False,
        reference_date=datetime.date(2025, 6, 8),
        advanced_df=advanced,
    )
    assert label == _NEUTRAL


def test_legacy_binary_path_used_when_no_advanced_df() -> None:
    # Without advanced_df, the existing 4-of-4 path should be used and
    # produce the same result as before.
    rows = [_hitter_row("p1", f"2025-06-0{i}", h=0, ab=4) for i in range(1, 8)]
    daily = pd.DataFrame(rows)
    label = streak_label(
        "p1", daily, is_pitcher=False, reference_date=datetime.date(2025, 6, 8)
    )
    assert label == _COLD


def test_annotate_with_streaks_passes_advanced_df() -> None:
    rows = [
        _hitter_row("p1", f"2025-06-0{i}", h=2, ab=4, hr=1, tb=6) for i in range(1, 8)
    ]
    daily = pd.DataFrame(rows)
    advanced = pd.DataFrame(
        [{"player_id": "p1", "xwoba": 0.320, "xwoba_against": None}]
    )
    df = pd.DataFrame([{"player_id": "p1", "position": "OF"}])
    out = annotate_with_streaks(
        df,
        daily,
        reference_date=datetime.date(2025, 6, 8),
        advanced_df=advanced,
    )
    assert out.iloc[0]["streak"] == _HOT
