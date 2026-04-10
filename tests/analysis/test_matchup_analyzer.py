"""
tests/analysis/test_matchup_analyzer.py

Tests for matchup_analyzer pure functions.
Uses realistic mid-week matchup fixture data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.matchup_analyzer import (
    _shrink_projection_rates,
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
    # days_remaining=1 → per-game rates × 1 = rates added directly
    result = project_week_totals(
        single_player_stats, single_player_proj, days_remaining=1
    )
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
    result = project_week_totals(stats, proj, days_remaining=1)
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
    result = project_week_totals(stats, proj, days_remaining=1)
    whip = result.iloc[0]["whip"]
    # (4+2+8+4) / (10+5) = 18/15 = 1.20
    expected = (4 + 2 + 8 + 4) / (10 + 5)
    assert whip == pytest.approx(expected, abs=1e-4), (
        f"Expected WHIP={expected}, got {whip}"
    )


# ---------------------------------------------------------------------------
# 3b. project_week_totals: per-game rates scaled by days_remaining
# ---------------------------------------------------------------------------


def test_project_week_totals_scales_by_days_remaining() -> None:
    # actual: H=5, proj per-game rate: H=2.0, days_remaining=3 → projected H = 5 + 2*3 = 11
    stats = pd.DataFrame([_make_stats_row("p1", h=5, ab=20)])
    proj = pd.DataFrame([_make_proj_row("p1", proj_h=2.0, proj_ab=6.0)])
    result = project_week_totals(stats, proj, days_remaining=3)
    assert result.iloc[0]["h"] == pytest.approx(11.0, abs=1e-6)
    assert result.iloc[0]["ab"] == pytest.approx(38.0, abs=1e-6)  # 20 + 6*3


def test_project_week_totals_zero_days_remaining_no_projection() -> None:
    # End of week: only actuals matter, projection adds nothing
    stats = pd.DataFrame([_make_stats_row("p1", h=10, ab=30)])
    proj = pd.DataFrame([_make_proj_row("p1", proj_h=5.0, proj_ab=15.0)])
    result = project_week_totals(stats, proj, days_remaining=0)
    assert result.iloc[0]["h"] == pytest.approx(10.0, abs=1e-6)
    assert result.iloc[0]["ab"] == pytest.approx(30.0, abs=1e-6)


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


# ---------------------------------------------------------------------------
# _shrink_projection_rates: Statcast-prior shrinkage of rate components
# ---------------------------------------------------------------------------


def test_shrink_projection_rates_none_or_empty_advanced_passes_through() -> None:
    stats = pd.DataFrame([_make_stats_row("p1", ab=40, bb=8, hbp=2)])
    proj = pd.DataFrame([_make_proj_row("p1", proj_hr=1.0, proj_k=6.0)])

    # Empty advanced_df: call returns projections unchanged.
    empty_adv = pd.DataFrame(
        columns=["player_id", "xwoba", "barrel_pct", "xwoba_against", "k_bb_pct"]
    )
    result = _shrink_projection_rates(proj, stats, empty_adv)
    assert result.iloc[0]["proj_hr"] == pytest.approx(1.0)
    assert result.iloc[0]["proj_k"] == pytest.approx(6.0)

    # project_week_totals with advanced_df=None also passes through.
    via_public = project_week_totals(stats, proj, days_remaining=1, advanced_df=None)
    assert len(via_public) == 1


def test_shrink_projection_rates_regresses_hot_hr_toward_barrel_prior() -> None:
    # Hot hitter pacing 1 HR/game but modest 10% Barrel — prior HR/PA = 0.06.
    # n_pa = 50; stabilization K = 170 → heavy pull toward prior.
    stats = pd.DataFrame([_make_stats_row("p1", ab=40, bb=8, hbp=2, sf=0)])
    proj = pd.DataFrame([_make_proj_row("p1", proj_hr=1.0)])
    adv = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "xwoba": 0.340,
                "barrel_pct": 10.0,
                "xwoba_against": None,
                "k_bb_pct": None,
            }
        ]
    )
    result = _shrink_projection_rates(proj, stats, adv)
    shrunk_hr = result.iloc[0]["proj_hr"]
    # Expected: (50 * (1.0/4.3) + 170 * 0.06) / 220 * 4.3 ≈ 0.4266
    assert shrunk_hr == pytest.approx(0.4266, abs=1e-3)
    # Other columns untouched.
    assert result.iloc[0]["proj_k"] == pytest.approx(5.0)


def test_shrink_projection_rates_pulls_low_k_up_toward_kbb_prior() -> None:
    # Pitcher with BF ≈ 100 (ip=20, ha=30, wa=10 → 60+30+10=100).
    # proj_k=4/game is below the K-BB% 20 → prior 0.28 K/BF implied level.
    stats = pd.DataFrame(
        [_make_stats_row("pit1", ip=20.0, hits_allowed=30, walks_allowed=10, k=20)]
    )
    proj = pd.DataFrame([_make_proj_row("pit1", proj_k=4.0)])
    adv = pd.DataFrame(
        [
            {
                "player_id": "pit1",
                "xwoba": None,
                "barrel_pct": None,
                "xwoba_against": None,
                "k_bb_pct": 20.0,
            }
        ]
    )
    result = _shrink_projection_rates(proj, stats, adv)
    shrunk_k = result.iloc[0]["proj_k"]
    # Expected: (100*(4/38.7) + 70*0.28)/170 * 38.7 ≈ 6.81
    assert shrunk_k == pytest.approx(6.81, abs=1e-2)
    # Observed is below prior → shrunk K should be higher than proj_k.
    assert shrunk_k > 4.0
    assert shrunk_k < 0.28 * 38.7  # but below the raw prior


def test_shrink_projection_rates_splits_walks_and_hits_proportionally() -> None:
    # BF = 100 (ip=20, ha=30, wa=10). proj_wa=3, proj_ha=7, combined=10/game.
    # xwoba_against=0.30 → WHIP prior=1.21 → prior_per_bf=1.21/4.3≈0.2814.
    stats = pd.DataFrame(
        [_make_stats_row("pit1", ip=20.0, hits_allowed=30, walks_allowed=10, k=20)]
    )
    proj = pd.DataFrame(
        [_make_proj_row("pit1", proj_walks_allowed=3.0, proj_hits_allowed=7.0)]
    )
    adv = pd.DataFrame(
        [
            {
                "player_id": "pit1",
                "xwoba": None,
                "barrel_pct": None,
                "xwoba_against": 0.30,
                "k_bb_pct": None,
            }
        ]
    )
    result = _shrink_projection_rates(proj, stats, adv)
    new_wa = result.iloc[0]["proj_walks_allowed"]
    new_ha = result.iloc[0]["proj_hits_allowed"]
    # shrunk_per_bf = (100*(10/38.7) + 770*(1.21/4.3))/870 ≈ 0.2787
    # shrunk_combined = 0.2787*38.7 ≈ 10.79
    combined = new_wa + new_ha
    assert combined == pytest.approx(10.79, abs=5e-2)
    # Split preserved: original 30% walks / 70% hits.
    assert new_wa / combined == pytest.approx(0.30, abs=1e-6)
    assert new_ha / combined == pytest.approx(0.70, abs=1e-6)


def test_shrink_projection_rates_missing_player_passes_through() -> None:
    # Player has no matching row in advanced_df → projection untouched.
    stats = pd.DataFrame([_make_stats_row("ghost", ab=40, bb=8)])
    proj = pd.DataFrame([_make_proj_row("ghost", proj_hr=0.8, proj_k=7.0)])
    adv = pd.DataFrame(
        [
            {
                "player_id": "someone_else",
                "xwoba": 0.340,
                "barrel_pct": 10.0,
                "xwoba_against": 0.30,
                "k_bb_pct": 20.0,
            }
        ]
    )
    result = _shrink_projection_rates(proj, stats, adv)
    assert result.iloc[0]["proj_hr"] == pytest.approx(0.8)
    assert result.iloc[0]["proj_k"] == pytest.approx(7.0)


def test_shrink_projection_rates_zero_pa_leaves_hr_unchanged() -> None:
    # No accumulated PA → no shrinkage weight → pass through.
    stats = pd.DataFrame([_make_stats_row("p1", ab=0, bb=0, hbp=0, sf=0)])
    proj = pd.DataFrame([_make_proj_row("p1", proj_hr=0.5)])
    adv = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "xwoba": 0.340,
                "barrel_pct": 10.0,
                "xwoba_against": None,
                "k_bb_pct": None,
            }
        ]
    )
    result = _shrink_projection_rates(proj, stats, adv)
    assert result.iloc[0]["proj_hr"] == pytest.approx(0.5)
