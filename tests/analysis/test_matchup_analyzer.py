"""
tests/analysis/test_matchup_analyzer.py

Tests for matchup_analyzer pure functions.
Uses realistic mid-week matchup fixture data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.matchup_analyzer import (
    check_ip_pace,
    get_focus_categories,
    project_week_totals,
    score_categories,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_stats_row(
    player_id: str,
    h: int = 5,
    ab: int = 20,
    hr: int = 1,
    sb: int = 1,
    bb: int = 2,
    hbp: int = 0,
    sf: int = 0,
    tb: int = 8,
    ip: float = 10.0,
    w: int = 1,
    k: int = 9,
    walks_allowed: int = 3,
    hits_allowed: int = 8,
    sv: int = 0,
    holds: int = 1,
    errors: int = 0,
    chances: int = 15,
) -> dict[str, object]:
    return {
        "player_id": player_id,
        "h": h,
        "ab": ab,
        "hr": hr,
        "sb": sb,
        "bb": bb,
        "hbp": hbp,
        "sf": sf,
        "tb": tb,
        "ip": ip,
        "w": w,
        "k": k,
        "walks_allowed": walks_allowed,
        "hits_allowed": hits_allowed,
        "sv": sv,
        "holds": holds,
        "errors": errors,
        "chances": chances,
        # Convenience columns (not used in computation)
        "avg": 0.250,
        "ops": 0.720,
        "fpct": 1.000,
        "whip": 1.10,
        "k_bb": 3.0,
        "sv_h": 1,
    }


def _make_proj_row(
    player_id: str,
    proj_h: float = 3.0,
    proj_ab: float = 12.0,
    proj_hr: float = 0.5,
    proj_sb: float = 0.5,
    proj_bb: float = 1.5,
    proj_hbp: float = 0.0,
    proj_sf: float = 0.0,
    proj_tb: float = 5.0,
    proj_ip: float = 5.0,
    proj_w: float = 0.5,
    proj_k: float = 5.0,
    proj_walks_allowed: float = 1.5,
    proj_hits_allowed: float = 4.0,
    proj_sv: float = 0.0,
    proj_holds: float = 0.5,
    proj_errors: float = 0.0,
    proj_chances: float = 7.0,
    games_remaining: int = 3,
) -> dict[str, object]:
    return {
        "player_id": player_id,
        "proj_h": proj_h,
        "proj_ab": proj_ab,
        "proj_hr": proj_hr,
        "proj_sb": proj_sb,
        "proj_bb": proj_bb,
        "proj_hbp": proj_hbp,
        "proj_sf": proj_sf,
        "proj_tb": proj_tb,
        "proj_ip": proj_ip,
        "proj_w": proj_w,
        "proj_k": proj_k,
        "proj_walks_allowed": proj_walks_allowed,
        "proj_hits_allowed": proj_hits_allowed,
        "proj_sv": proj_sv,
        "proj_holds": proj_holds,
        "proj_errors": proj_errors,
        "proj_chances": proj_chances,
        "games_remaining": games_remaining,
        "proj_avg": 0.250,
        "proj_ops": 0.720,
        "proj_fpct": 1.000,
        "proj_whip": 1.10,
        "proj_k_bb": 3.0,
        "source": "steamer",
        "projection_date": "2026-03-14",
        "target_week": 1,
    }


@pytest.fixture()
def single_player_stats() -> pd.DataFrame:
    return pd.DataFrame([_make_stats_row("p1", h=5, ab=20, tb=8)])


@pytest.fixture()
def single_player_proj() -> pd.DataFrame:
    return pd.DataFrame([_make_proj_row("p1", proj_h=3.0, proj_ab=12.0, proj_tb=5.0)])


# ---------------------------------------------------------------------------
# 1. project_week_totals: counting stats add correctly
# ---------------------------------------------------------------------------


def test_project_week_totals_adds_counting_stats(
    single_player_stats: pd.DataFrame,
    single_player_proj: pd.DataFrame,
) -> None:
    result = project_week_totals(single_player_stats, single_player_proj)
    assert len(result) == 1
    total_h = result.iloc[0]["h"]
    assert total_h == pytest.approx(8.0, abs=1e-6), f"Expected H=8, got {total_h}"


# ---------------------------------------------------------------------------
# 2. project_week_totals: AVG uses components, not averaging
# ---------------------------------------------------------------------------


def test_project_week_totals_rate_stat_uses_components() -> None:
    # Use numbers where straight averaging diverges from component-based computation.
    # stats: H=10, AB=20 → naive avg=0.500
    # proj:  H=1, AB=10  → naive avg=0.100
    # Naive average of rates: (0.500 + 0.100) / 2 = 0.300
    # Component AVG: (10+1)/(20+10) = 11/30 ≈ 0.3667 (differs from naive average)
    stats = pd.DataFrame([_make_stats_row("p1", h=10, ab=20)])
    proj = pd.DataFrame([_make_proj_row("p1", proj_h=1.0, proj_ab=10.0)])
    result = project_week_totals(stats, proj)
    avg = result.iloc[0]["avg"]
    assert avg == pytest.approx(11 / 30, abs=1e-4), f"Expected AVG=11/30, got {avg}"


# ---------------------------------------------------------------------------
# 3. project_week_totals: WHIP uses components
# ---------------------------------------------------------------------------


def test_project_week_totals_whip_uses_components() -> None:
    stats = pd.DataFrame(
        [_make_stats_row("p1", ip=10.0, walks_allowed=4, hits_allowed=8)]
    )
    proj = pd.DataFrame(
        [
            _make_proj_row(
                "p1", proj_ip=5.0, proj_walks_allowed=2.0, proj_hits_allowed=4.0
            )
        ]
    )
    result = project_week_totals(stats, proj)
    whip = result.iloc[0]["whip"]
    # (4+2+8+4) / (10+5) = 18/15 = 1.20
    expected = (4 + 2 + 8 + 4) / (10 + 5)
    assert whip == pytest.approx(expected, abs=1e-4), (
        f"Expected WHIP={expected}, got {whip}"
    )


# ---------------------------------------------------------------------------
# 4. score_categories: highest-wins category gives my_leads=True when I'm ahead
# ---------------------------------------------------------------------------


def test_score_categories_highest_wins() -> None:
    my_totals = pd.DataFrame([{"h": 30, "hr": 5}])
    opp_totals = pd.DataFrame([{"h": 20, "hr": 3}])
    config = {"h": "highest", "hr": "highest"}
    result = score_categories(my_totals, opp_totals, config)

    h_row = result[result["category"] == "h"].iloc[0]
    assert bool(h_row["my_leads"]) is True
    assert h_row["my_value"] == pytest.approx(30.0)
    assert h_row["opp_value"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# 5. score_categories: WHIP inverted — lower is better
# ---------------------------------------------------------------------------


def test_score_categories_whip_inverted() -> None:
    # My WHIP is lower (better)
    my_totals = pd.DataFrame([{"whip": 1.10}])
    opp_totals = pd.DataFrame([{"whip": 1.40}])
    config = {"whip": "lowest"}
    result = score_categories(my_totals, opp_totals, config)

    whip_row = result[result["category"] == "whip"].iloc[0]
    assert bool(whip_row["my_leads"]) is True


# ---------------------------------------------------------------------------
# 6. score_categories: status classification thresholds
# ---------------------------------------------------------------------------


def test_score_categories_status_classification() -> None:
    # >15% margin → safe
    # 5-15% → flippable
    # <5% → toss_up
    my_totals = pd.DataFrame(
        [
            {
                "h": 100,  # 25% lead → safe_win
                "hr": 21,  # ~10% lead → flippable_win (21 vs 19: 2/21 ≈ 9.5%)
                "sb": 10,  # 2% lead → toss_up (10 vs 9.8)
                "bb": 0,  # tied → toss_up
            }
        ]
    )
    opp_totals = pd.DataFrame(
        [
            {
                "h": 80,
                "hr": 19,
                "sb": 9.8,
                "bb": 0,
            }
        ]
    )
    config = {"h": "highest", "hr": "highest", "sb": "highest", "bb": "highest"}
    result = score_categories(my_totals, opp_totals, config)

    def get_status(cat: str) -> str:
        return str(result[result["category"] == cat].iloc[0]["status"])

    # H: margin = |100-80|/100 = 0.20 ≥ 0.15 → safe_win
    assert get_status("h") == "safe_win", f"Got: {get_status('h')}"
    # HR: margin = |21-19|/21 ≈ 0.095 → flippable_win
    assert get_status("hr") == "flippable_win", f"Got: {get_status('hr')}"
    # SB: margin = |10-9.8|/10 = 0.02 < 0.05 → toss_up
    assert get_status("sb") == "toss_up", f"Got: {get_status('sb')}"
    # BB: tied → toss_up
    assert get_status("bb") == "toss_up", f"Got: {get_status('bb')}"


# ---------------------------------------------------------------------------
# 7. get_focus_categories: returns flippable and toss_up only
# ---------------------------------------------------------------------------


def test_get_focus_categories_returns_flippable_and_tossup() -> None:
    scored_df = pd.DataFrame(
        [
            {"category": "h", "status": "safe_win"},
            {"category": "hr", "status": "flippable_win"},
            {"category": "sb", "status": "toss_up"},
            {"category": "bb", "status": "flippable_loss"},
            {"category": "whip", "status": "safe_loss"},
        ]
    )
    focus = get_focus_categories(scored_df)
    assert set(focus) == {"hr", "sb", "bb"}
    assert "h" not in focus
    assert "whip" not in focus


# ---------------------------------------------------------------------------
# 8. check_ip_pace: on pace
# ---------------------------------------------------------------------------


def test_check_ip_pace_on_pace() -> None:
    # 15 IP in 3 days elapsed (7-4=3 days elapsed), 4 days remaining
    # projected = 15 + (15/3)*4 = 15 + 20 = 35 → on pace
    stats = pd.DataFrame(
        [{"player_id": "p1", "ip": 9.0}, {"player_id": "p2", "ip": 6.0}]
    )
    result = check_ip_pace(stats, days_remaining=4, min_ip=21)
    assert result["current_ip"] == pytest.approx(15.0)
    assert result["projected_ip"] == pytest.approx(35.0, abs=1e-3)
    assert result["on_pace"] is True


# ---------------------------------------------------------------------------
# 9. check_ip_pace: not on pace
# ---------------------------------------------------------------------------


def test_check_ip_pace_not_on_pace() -> None:
    # 8 IP in 6 days elapsed (7-1=6), 1 day remaining
    # projected = 8 + (8/6)*1 ≈ 8 + 1.333 = 9.333 → not on pace
    stats = pd.DataFrame([{"player_id": "p1", "ip": 8.0}])
    result = check_ip_pace(stats, days_remaining=1, min_ip=21)
    assert result["current_ip"] == pytest.approx(8.0)
    assert result["projected_ip"] == pytest.approx(8 + 8 / 6, abs=1e-3)
    assert result["on_pace"] is False
