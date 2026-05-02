"""
tests/analysis/test_waiver_ranker.py

Tests for waiver_ranker pure functions.
Uses realistic mid-week matchup fixture data.
"""

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from src.analysis.waiver_ranker import (
    _ALPHA_EARLY,
    _ALPHA_LATE,
    alpha_from_season_progress,
    find_recommended_drop,
    rank_free_agents,
    score_free_agent,
)
from src.config import load_league_settings

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> object:
    return load_league_settings()


def _make_matchup_df(
    status_by_cat: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Create a sample scored matchup DataFrame."""
    if status_by_cat is None:
        status_by_cat = {
            "h": "safe_win",
            "hr": "flippable_win",
            "sb": "toss_up",
            "bb": "flippable_loss",
            "fpct": "safe_win",
            "avg": "flippable_win",
            "ops": "toss_up",
            "w": "safe_loss",
            "k": "flippable_loss",
            "whip": "toss_up",
            "k_bb": "safe_win",
            "sv_h": "flippable_win",
        }
    records = []
    for cat, status in status_by_cat.items():
        records.append(
            {
                "category": cat,
                "my_value": 1.0,
                "opp_value": 1.0,
                "my_leads": True,
                "margin_pct": 0.1,
                "win_prob": 0.7,
                "status": status,
            }
        )
    return pd.DataFrame(records)


def _make_player_row(
    player_id: str = "fa1",
    h: float = 3.0,
    hr: float = 1.0,
    sb: float = 0.5,
    bb: float = 1.0,
    fpct: float = 0.990,
    avg: float = 0.280,
    ops: float = 0.820,
    w: float = 0.0,
    k: float = 0.0,
    whip: float = 0.0,
    k_bb: float = 0.0,
    sv_h: float = 0.0,
    eligible_positions: list[str] | None = None,
    slot: str = "BN",
    overall_score: float = 5.0,
    games_remaining: int = 3,
) -> pd.Series:
    if eligible_positions is None:
        eligible_positions = ["OF"]
    return pd.Series(
        {
            "player_id": player_id,
            "h": h,
            "hr": hr,
            "sb": sb,
            "bb": bb,
            "fpct": fpct,
            "avg": avg,
            "ops": ops,
            "w": w,
            "k": k,
            "whip": whip,
            "k_bb": k_bb,
            "sv_h": sv_h,
            "eligible_positions": eligible_positions,
            "slot": slot,
            "overall_score": overall_score,
            "games_remaining": games_remaining,
        }
    )


def _make_roster_df() -> pd.DataFrame:
    """Return a realistic 16-player active roster."""
    rows = [
        # Position players
        _make_player_row("c1", eligible_positions=["C"], slot="C", overall_score=6.0),
        _make_player_row(
            "1b1", eligible_positions=["1B"], slot="1B", overall_score=7.0
        ),
        _make_player_row(
            "2b1", eligible_positions=["2B"], slot="2B", overall_score=5.5
        ),
        _make_player_row(
            "3b1", eligible_positions=["3B"], slot="3B", overall_score=5.0
        ),
        _make_player_row(
            "ss1", eligible_positions=["SS"], slot="SS", overall_score=5.0
        ),
        _make_player_row(
            "of1", eligible_positions=["OF"], slot="OF", overall_score=8.0
        ),
        _make_player_row(
            "of2", eligible_positions=["OF"], slot="OF", overall_score=7.5
        ),
        _make_player_row(
            "of3", eligible_positions=["OF"], slot="OF", overall_score=7.0
        ),
        _make_player_row(
            "util1", eligible_positions=["1B", "OF"], slot="Util", overall_score=6.0
        ),
        _make_player_row(
            "util2", eligible_positions=["2B", "SS"], slot="Util", overall_score=4.0
        ),
        # Pitchers
        _make_player_row(
            "sp1",
            eligible_positions=["SP"],
            slot="SP",
            w=1.0,
            k=8.0,
            whip=1.15,
            k_bb=3.2,
            sv_h=0.0,
            overall_score=7.0,
        ),
        _make_player_row(
            "sp2",
            eligible_positions=["SP"],
            slot="SP",
            w=0.5,
            k=6.0,
            whip=1.30,
            k_bb=2.5,
            sv_h=0.0,
            overall_score=5.0,
        ),
        _make_player_row(
            "rp1",
            eligible_positions=["RP"],
            slot="RP",
            sv_h=2.0,
            k=3.0,
            whip=1.10,
            k_bb=2.0,
            overall_score=4.0,
        ),
        _make_player_row(
            "rp2",
            eligible_positions=["RP"],
            slot="RP",
            sv_h=1.5,
            k=2.0,
            whip=1.20,
            k_bb=1.8,
            overall_score=3.5,
        ),
        # Bench players (BN)
        _make_player_row(
            "bn1",
            eligible_positions=["OF"],
            slot="BN",
            overall_score=2.0,
            games_remaining=0,
        ),
        _make_player_row(
            "bn2",
            eligible_positions=["1B", "OF"],
            slot="BN",
            overall_score=3.0,
            games_remaining=2,
        ),
    ]
    return pd.DataFrame([r.to_dict() for r in rows])


# ---------------------------------------------------------------------------
# 1. score_free_agent: flippable categories weighted higher than safe ones
# ---------------------------------------------------------------------------


def test_score_free_agent_weights_flippable_higher(config: object) -> None:
    from src.config import LeagueSettings

    assert isinstance(config, LeagueSettings)
    roster = _make_roster_df()

    # Create a matchup where "hr" is flippable and "h" is safe_win
    matchup = _make_matchup_df(
        {
            "h": "safe_win",
            "hr": "flippable_loss",
            "sb": "safe_win",
            "bb": "safe_win",
            "fpct": "safe_win",
            "avg": "safe_win",
            "ops": "safe_win",
            "w": "safe_win",
            "k": "safe_win",
            "whip": "safe_win",
            "k_bb": "safe_win",
            "sv_h": "safe_win",
        }
    )

    # Player A: big HR producer, few H
    player_a = _make_player_row("fa_hr", hr=5.0, h=1.0)
    # Player B: big H producer, few HR
    player_b = _make_player_row("fa_h", hr=0.0, h=8.0)

    # We need a drop candidate — use bn1 which has low overall_score
    # Both players compare against bn1 (overall_score=2.0, all stats ~0)

    score_a = score_free_agent(player_a, roster, matchup, config)
    score_b = score_free_agent(player_b, roster, matchup, config)

    # Player A should score higher because HR is in a flippable category (weight 2.0)
    # vs H in safe_win (weight 0.1)
    overall_a = cast(float, score_a["overall_score"])
    overall_b = cast(float, score_b["overall_score"])
    assert overall_a > overall_b, (
        f"Expected HR producer to outscore H producer when HR is flippable. "
        f"score_a={overall_a:.3f}, score_b={overall_b:.3f}"
    )


# ---------------------------------------------------------------------------
# 2. score_free_agent: lower WHIP scores positively
# ---------------------------------------------------------------------------


def test_score_free_agent_whip_direction(config: object) -> None:
    from src.config import LeagueSettings

    assert isinstance(config, LeagueSettings)

    # Build a roster with NO bench players so the drop must come from active pitchers.
    # rp2 has whip=2.0 and lowest overall_score → will be the recommended drop.
    roster_no_bench = pd.DataFrame(
        [
            _make_player_row(
                "sp1",
                eligible_positions=["SP"],
                slot="SP",
                overall_score=7.0,
                whip=1.15,
            ).to_dict(),
            _make_player_row(
                "rp1",
                eligible_positions=["RP"],
                slot="RP",
                overall_score=4.0,
                whip=1.10,
            ).to_dict(),
            _make_player_row(
                "rp2", eligible_positions=["RP"], slot="RP", overall_score=0.5, whip=2.0
            ).to_dict(),
        ]
    )

    matchup = _make_matchup_df(
        {
            "h": "safe_win",
            "hr": "safe_win",
            "sb": "safe_win",
            "bb": "safe_win",
            "fpct": "safe_win",
            "avg": "safe_win",
            "ops": "safe_win",
            "w": "safe_win",
            "k": "safe_win",
            "whip": "toss_up",  # WHIP is contested
            "k_bb": "safe_win",
            "sv_h": "safe_win",
        }
    )

    # A pitcher with excellent WHIP (0.90) replaces rp2 (whip=2.0, lowest score)
    good_pitcher = _make_player_row(
        "fa_pitcher",
        whip=0.90,
        k=7.0,
        sv_h=1.0,
        eligible_positions=["RP"],
    )
    result = score_free_agent(good_pitcher, roster_no_bench, matchup, config)

    # find_recommended_drop should pick rp2 (lowest overall_score, RP position match).
    assert result["recommended_drop_id"] == "rp2", (
        f"Expected drop rp2 but got {result['recommended_drop_id']}"
    )

    # Because WHIP direction is "lowest" and good_pitcher.whip (0.90) < rp2.whip (2.0),
    # delta = 0.90 - 2.0 = -1.10, inverted → +1.10 contribution → positive score for WHIP
    category_scores = cast(dict[str, float], result["category_scores"])
    whip_score = category_scores.get("whip", 0.0)
    assert isinstance(whip_score, float)
    assert whip_score > 0, f"Expected positive WHIP contribution, got {whip_score}"


# ---------------------------------------------------------------------------
# 3. find_recommended_drop: never drop the only C
# ---------------------------------------------------------------------------


def test_find_recommended_drop_respects_position(config: object) -> None:
    from src.config import LeagueSettings

    assert isinstance(config, LeagueSettings)
    roster = _make_roster_df()

    # Candidate is a Catcher replacement — but we already have the only C
    # The only C is "c1" — we should not recommend dropping them
    candidate = _make_player_row("new_c", eligible_positions=["C"])

    # Make the C player have the lowest overall_score to test that we don't drop them
    idx = roster[roster["player_id"] == "c1"].index
    roster.loc[idx, "overall_score"] = 0.1  # lowest score ever

    drop_id = find_recommended_drop(candidate, roster, config)

    # The recommended drop should NOT be c1 since there's a BN player to drop instead
    assert drop_id != "c1", f"Should not drop the only Catcher, got {drop_id}"


# ---------------------------------------------------------------------------
# 4. find_recommended_drop: prefers bench over active starter
# ---------------------------------------------------------------------------


def test_find_recommended_drop_prefers_bench_over_starter(config: object) -> None:
    from src.config import LeagueSettings

    assert isinstance(config, LeagueSettings)
    roster = _make_roster_df()

    # OF candidate — both OF active starters and BN players are position-eligible
    candidate = _make_player_row("new_of", eligible_positions=["OF"])

    drop_id = find_recommended_drop(candidate, roster, config)

    # bn1 has slot=BN and games_remaining=0, lowest overall_score on bench
    # Should prefer dropping BN player over active OF
    assert drop_id == "bn1", f"Expected drop bn1 (bench, no games), got {drop_id}"


# ---------------------------------------------------------------------------
# 5. rank_free_agents: sorted descending by overall_score
# ---------------------------------------------------------------------------


def test_rank_free_agents_sorted_descending(config: object) -> None:
    from src.config import LeagueSettings

    assert isinstance(config, LeagueSettings)
    roster = _make_roster_df()
    matchup = _make_matchup_df()
    callups = pd.DataFrame(columns=["player_id", "days_since_callup"])

    # Three free agents with varying stats — all OF eligible
    free_agents = pd.DataFrame(
        [
            _make_player_row("fa1", hr=3.0, sb=2.0, h=5.0).to_dict(),
            _make_player_row("fa2", hr=0.5, sb=0.5, h=2.0).to_dict(),
            _make_player_row("fa3", hr=2.0, sb=1.5, h=4.0).to_dict(),
        ]
    )

    result = rank_free_agents(free_agents, roster, matchup, callups, config)

    assert len(result) == 3
    scores = result["overall_score"].tolist()
    assert scores == sorted(scores, reverse=True), f"Not sorted descending: {scores}"


# ---------------------------------------------------------------------------
# 6. rank_free_agents: call-up players are flagged
# ---------------------------------------------------------------------------


def test_rank_free_agents_marks_callups(config: object) -> None:
    from src.config import LeagueSettings

    assert isinstance(config, LeagueSettings)
    roster = _make_roster_df()
    matchup = _make_matchup_df()

    free_agents = pd.DataFrame(
        [
            _make_player_row("fa_callup", hr=2.0, h=4.0).to_dict(),
            _make_player_row("fa_normal", hr=1.0, h=3.0).to_dict(),
        ]
    )

    callups = pd.DataFrame([{"player_id": "fa_callup", "days_since_callup": 2}])

    result = rank_free_agents(free_agents, roster, matchup, callups, config)

    callup_row = result[result["player_id"] == "fa_callup"].iloc[0]
    normal_row = result[result["player_id"] == "fa_normal"].iloc[0]

    assert bool(callup_row["is_callup"]) is True
    assert callup_row["days_since_callup"] == 2
    assert bool(normal_row["is_callup"]) is False


# ---------------------------------------------------------------------------
# 7. rank_free_agents: output carries display metadata (name/team/position)
# ---------------------------------------------------------------------------


def test_rank_free_agents_preserves_metadata(config: object) -> None:
    from src.config import LeagueSettings

    assert isinstance(config, LeagueSettings)
    roster = _make_roster_df()
    matchup = _make_matchup_df()
    callups = pd.DataFrame(columns=["player_id", "days_since_callup"])

    fa_row = _make_player_row("fa1", hr=2.0).to_dict()
    fa_row["full_name"] = "Fresh Face"
    fa_row["team"] = "NYY"
    # Free agent df uses `positions` (Yahoo schema), not `eligible_positions`.
    fa_row["positions"] = ["OF"]
    free_agents = pd.DataFrame([fa_row])

    result = rank_free_agents(free_agents, roster, matchup, callups, config)

    assert not result.empty
    row = result.iloc[0]
    assert row["player_name"] == "Fresh Face"
    assert row["team"] == "NYY"
    assert "OF" in str(row["position"])
    assert bool(row["is_pitcher"]) is False
    assert "fit_score" in result.columns
    # Carries per-game stats for display
    assert "hr" in result.columns


# ---------------------------------------------------------------------------
# 8. score_free_agent: pitchers don't get penalized for 0 hitting stats
# ---------------------------------------------------------------------------


def test_score_free_agent_pitcher_skips_hitting_cats(config: object) -> None:
    from src.config import LeagueSettings

    assert isinstance(config, LeagueSettings)
    roster = _make_roster_df()
    matchup = _make_matchup_df()

    pitcher = _make_player_row(
        "fa_pitcher",
        eligible_positions=["SP"],
        h=0.0,
        hr=0.0,
        sb=0.0,
        bb=0.0,
        w=1.0,
        k=7.0,
        whip=1.05,
        k_bb=3.0,
        sv_h=0.0,
    )

    result = score_free_agent(pitcher, roster, matchup, config)
    cat_scores = cast(dict[str, float], result["category_scores"])

    # Hitting categories should all be 0 (pitcher skips them)
    for hit_cat in ("h", "hr", "sb", "bb", "avg", "ops", "fpct"):
        assert cat_scores.get(hit_cat, 0.0) == 0.0


# ---------------------------------------------------------------------------
# 9. Talent vs Fit decomposition + α blend
# ---------------------------------------------------------------------------


def test_score_free_agent_returns_talent_and_fit_components(config: object) -> None:
    from src.config import LeagueSettings

    assert isinstance(config, LeagueSettings)
    roster = _make_roster_df()
    matchup = _make_matchup_df()
    player = _make_player_row("fa1", hr=2.0, h=4.0)

    result = score_free_agent(player, roster, matchup, config)

    assert "talent_score" in result
    assert "fit_score" in result
    assert "weighted_score" in result
    assert "overall_score" in result
    # Legacy contract: overall_score from a single-player call equals weighted_score.
    assert result["overall_score"] == result["weighted_score"]


def test_alpha_decays_with_season_progress() -> None:
    early = alpha_from_season_progress(0.0)
    mid = alpha_from_season_progress(0.5)
    late = alpha_from_season_progress(1.0)
    assert early == pytest.approx(_ALPHA_EARLY)
    assert late == pytest.approx(_ALPHA_LATE)
    assert early > mid > late
    # Out-of-range clamping
    assert alpha_from_season_progress(-0.5) == pytest.approx(_ALPHA_EARLY)
    assert alpha_from_season_progress(1.5) == pytest.approx(_ALPHA_LATE)


def test_rank_free_agents_blends_talent_and_fit(config: object) -> None:
    from src.config import LeagueSettings

    assert isinstance(config, LeagueSettings)
    roster = _make_roster_df()
    callups = pd.DataFrame(columns=["player_id", "days_since_callup"])

    # One category flippable so fit_score has a clear winner
    matchup = _make_matchup_df(
        {
            "h": "safe_win",
            "hr": "flippable_loss",
            "sb": "safe_win",
            "bb": "safe_win",
            "fpct": "safe_win",
            "avg": "safe_win",
            "ops": "safe_win",
            "w": "safe_win",
            "k": "safe_win",
            "whip": "safe_win",
            "k_bb": "safe_win",
            "sv_h": "safe_win",
        }
    )

    fa_hr = _make_player_row("fa_hr", hr=4.0, h=1.0).to_dict()
    fa_h = _make_player_row("fa_h", hr=0.0, h=8.0).to_dict()
    free_agents = pd.DataFrame([fa_hr, fa_h])

    early = rank_free_agents(
        free_agents, roster, matchup, callups, config, season_progress=0.0
    )
    late = rank_free_agents(
        free_agents, roster, matchup, callups, config, season_progress=1.0
    )

    # Pull each player's overall_score in early vs late
    early_hr = float(early.loc[early["player_id"] == "fa_hr", "overall_score"].iloc[0])
    early_h = float(early.loc[early["player_id"] == "fa_h", "overall_score"].iloc[0])
    late_hr = float(late.loc[late["player_id"] == "fa_hr", "overall_score"].iloc[0])
    late_h = float(late.loc[late["player_id"] == "fa_h", "overall_score"].iloc[0])

    # Late season weights fit higher → fa_hr's flippable HR advantage matters more.
    # The fa_hr − fa_h gap should be larger late season than early season.
    early_gap = early_hr - early_h
    late_gap = late_hr - late_h
    assert late_gap > early_gap, (
        f"Late-season gap should exceed early-season gap. "
        f"early_gap={early_gap:.3f}, late_gap={late_gap:.3f}"
    )


def test_rank_free_agents_exposes_components(config: object) -> None:
    from src.config import LeagueSettings

    assert isinstance(config, LeagueSettings)
    roster = _make_roster_df()
    matchup = _make_matchup_df()
    callups = pd.DataFrame(columns=["player_id", "days_since_callup"])

    free_agents = pd.DataFrame([_make_player_row("fa1", hr=2.0).to_dict()])
    result = rank_free_agents(free_agents, roster, matchup, callups, config)

    assert "talent_score" in result.columns
    assert "fit_score" in result.columns
    assert "weighted_score" in result.columns
    assert "overall_score" in result.columns


# ---------------------------------------------------------------------------
# 10. find_recommended_drop: pitcher/hitter type enforcement
# ---------------------------------------------------------------------------


def test_find_recommended_drop_enforces_same_type(config: object) -> None:
    """Adding a hitter should never recommend dropping a pitcher."""
    from src.config import LeagueSettings

    assert isinstance(config, LeagueSettings)

    # Roster with a low-score pitcher and a higher-score hitter on bench
    roster = pd.DataFrame(
        [
            _make_player_row(
                "sp_low",
                eligible_positions=["SP"],
                slot="BN",
                overall_score=1.0,
                w=0.5,
                k=4.0,
                whip=1.5,
            ).to_dict(),
            _make_player_row(
                "of_bench",
                eligible_positions=["OF"],
                slot="BN",
                overall_score=3.0,
            ).to_dict(),
            _make_player_row(
                "of_active",
                eligible_positions=["OF"],
                slot="OF",
                overall_score=7.0,
            ).to_dict(),
        ]
    )

    # Candidate is an OF (hitter) — should NOT recommend dropping sp_low
    candidate = _make_player_row("new_of", eligible_positions=["OF"])
    drop_id = find_recommended_drop(candidate, roster, config)

    # Must drop a hitter, not the pitcher
    assert drop_id == "of_bench", f"Expected of_bench drop, got {drop_id}"


def test_find_recommended_drop_pitcher_for_pitcher(config: object) -> None:
    """Adding a pitcher should recommend dropping a pitcher, not a hitter."""
    from src.config import LeagueSettings

    assert isinstance(config, LeagueSettings)

    roster = pd.DataFrame(
        [
            _make_player_row(
                "rp_low",
                eligible_positions=["RP"],
                slot="BN",
                overall_score=1.0,
                sv_h=0.5,
                k=2.0,
                whip=1.6,
            ).to_dict(),
            _make_player_row(
                "of_bench",
                eligible_positions=["OF"],
                slot="BN",
                overall_score=0.5,  # lower than rp_low
            ).to_dict(),
        ]
    )

    # Candidate is an RP (pitcher)
    candidate = _make_player_row("new_rp", eligible_positions=["RP"])
    drop_id = find_recommended_drop(candidate, roster, config)

    # Must drop the pitcher, even though the OF has a lower score
    assert drop_id == "rp_low", f"Expected rp_low drop, got {drop_id}"


# ---------------------------------------------------------------------------
# 11. find_recommended_drop: exclude_ids prevents reuse
# ---------------------------------------------------------------------------


def test_find_recommended_drop_respects_exclude_ids(config: object) -> None:
    """Excluding an ID forces the next-best drop candidate."""
    from src.config import LeagueSettings

    assert isinstance(config, LeagueSettings)

    roster = pd.DataFrame(
        [
            _make_player_row(
                "bn1", eligible_positions=["OF"], slot="BN", overall_score=1.0
            ).to_dict(),
            _make_player_row(
                "bn2", eligible_positions=["OF"], slot="BN", overall_score=2.0
            ).to_dict(),
        ]
    )

    candidate = _make_player_row("new_of", eligible_positions=["OF"])

    # Without exclusion → picks bn1 (lowest score)
    drop1 = find_recommended_drop(candidate, roster, config)
    assert drop1 == "bn1"

    # Exclude bn1 → picks bn2
    drop2 = find_recommended_drop(candidate, roster, config, exclude_ids={"bn1"})
    assert drop2 == "bn2"


# ---------------------------------------------------------------------------
# 12. Positional need: stacked position gets penalized in rankings
# ---------------------------------------------------------------------------


def test_rank_free_agents_penalizes_stacked_position(config: object) -> None:
    """A catcher FA should be penalized when roster already has a strong catcher."""
    from src.config import LeagueSettings

    assert isinstance(config, LeagueSettings)

    # Roster with a strong catcher but weak OF (below median).
    # Median will be driven by the mix — C at 9.0 is above median, OF at 2.0 below.
    roster = pd.DataFrame(
        [
            _make_player_row(
                "c1",
                eligible_positions=["C"],
                slot="C",
                overall_score=9.0,
                h=4.0,
                hr=2.0,
            ).to_dict(),
            _make_player_row(
                "of1",
                eligible_positions=["OF"],
                slot="OF",
                overall_score=2.0,
                h=1.0,
                hr=0.5,
            ).to_dict(),
            _make_player_row(
                "of2",
                eligible_positions=["OF"],
                slot="OF",
                overall_score=2.0,
                h=1.5,
                hr=0.5,
            ).to_dict(),
            _make_player_row(
                "of3",
                eligible_positions=["OF"],
                slot="OF",
                overall_score=2.0,
                h=1.0,
                hr=0.5,
            ).to_dict(),
            _make_player_row(
                "sp1",
                eligible_positions=["SP"],
                slot="SP",
                overall_score=7.0,
                w=1.0,
                k=7.0,
                whip=1.1,
            ).to_dict(),
            _make_player_row(
                "bn1",
                eligible_positions=["OF"],
                slot="BN",
                overall_score=1.0,
            ).to_dict(),
        ]
    )
    matchup = _make_matchup_df()
    callups = pd.DataFrame(columns=["player_id", "days_since_callup"])

    # Two FAs with identical raw stats: one C, one OF
    fa_c = _make_player_row("fa_c", eligible_positions=["C"], hr=2.0, h=3.0).to_dict()
    fa_of = _make_player_row(
        "fa_of", eligible_positions=["OF"], hr=2.0, h=3.0
    ).to_dict()
    free_agents = pd.DataFrame([fa_c, fa_of])

    result = rank_free_agents(free_agents, roster, matchup, callups, config)

    c_score = float(result.loc[result["player_id"] == "fa_c", "overall_score"].iloc[0])
    of_score = float(
        result.loc[result["player_id"] == "fa_of", "overall_score"].iloc[0]
    )

    # C should be penalized (already stacked), OF should rank higher
    assert of_score > c_score, (
        f"Expected OF FA to rank higher than C FA when C is stacked. "
        f"OF={of_score:.3f}, C={c_score:.3f}"
    )
