"""
tests/analysis/test_lineup_optimizer.py

Tests for lineup_optimizer pure functions.
Uses realistic mid-week matchup fixture data.
"""

from __future__ import annotations

import datetime
import json
from typing import cast

import pandas as pd
import pytest

from src.analysis.lineup_optimizer import (
    build_daily_report,
    optimize_daily_lineup,
    recommend_adds,
)
from src.config import load_league_settings

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> object:
    return load_league_settings()


def _make_matchup_df(
    statuses: dict[str, str] | None = None,
) -> pd.DataFrame:
    if statuses is None:
        statuses = {
            "h": "safe_win",
            "hr": "flippable_win",
            "sb": "toss_up",
            "bb": "flippable_loss",
            "fpct": "safe_win",
            "avg": "flippable_win",
            "ops": "safe_win",
            "w": "safe_loss",
            "k": "flippable_loss",
            "whip": "toss_up",
            "k_bb": "safe_win",
            "sv_h": "flippable_win",
        }
    records = []
    for cat, status in statuses.items():
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


def _build_roster_df(include_no_game_player: bool = True) -> pd.DataFrame:
    """Build a 26-slot roster DataFrame for testing."""
    rows = [
        # Active starters
        {
            "player_id": "c1",
            "slot": "C",
            "eligible_positions": ["C"],
            "games_today": True,
            "accumulated_ip": 0.0,
            "sb": 0.0,
            "hr": 0.0,
        },
        {
            "player_id": "1b1",
            "slot": "1B",
            "eligible_positions": ["1B"],
            "games_today": True,
            "accumulated_ip": 0.0,
            "sb": 0.0,
            "hr": 2.0,
        },
        {
            "player_id": "2b1",
            "slot": "2B",
            "eligible_positions": ["2B"],
            "games_today": True,
            "accumulated_ip": 0.0,
            "sb": 1.0,
            "hr": 0.0,
        },
        {
            "player_id": "3b1",
            "slot": "3B",
            "eligible_positions": ["3B"],
            "games_today": True,
            "accumulated_ip": 0.0,
            "sb": 0.0,
            "hr": 1.0,
        },
        {
            "player_id": "ss1",
            "slot": "SS",
            "eligible_positions": ["SS"],
            "games_today": True,
            "accumulated_ip": 0.0,
            "sb": 2.0,
            "hr": 0.0,
        },
        {
            "player_id": "of1",
            "slot": "OF",
            "eligible_positions": ["OF"],
            "games_today": True,
            "accumulated_ip": 0.0,
            "sb": 3.0,
            "hr": 1.0,
        },
        {
            "player_id": "of2",
            "slot": "OF",
            "eligible_positions": ["OF"],
            "games_today": True,
            "accumulated_ip": 0.0,
            "sb": 1.0,
            "hr": 2.0,
        },
        {
            "player_id": "of3",
            "slot": "OF",
            "eligible_positions": ["OF"],
            "games_today": True,
            "accumulated_ip": 0.0,
            "sb": 0.0,
            "hr": 0.0,
        },
        # Util1: batter eligible for Util with good SB (flippable)
        {
            "player_id": "util1",
            "slot": "Util",
            "eligible_positions": ["1B", "OF"],
            "games_today": True,
            "accumulated_ip": 0.0,
            "sb": 5.0,  # big SB contributor
            "hr": 0.5,
        },
        # Util2: batter eligible for Util
        {
            "player_id": "util2",
            "slot": "Util",
            "eligible_positions": ["2B", "SS"],
            "games_today": True,
            "accumulated_ip": 0.0,
            "sb": 0.0,
            "hr": 0.0,
        },
        # Pitchers
        {
            "player_id": "sp1",
            "slot": "SP",
            "eligible_positions": ["SP"],
            "games_today": True,
            "accumulated_ip": 6.0,
            "sb": 0.0,
            "hr": 0.0,
        },
        {
            "player_id": "sp2",
            "slot": "SP",
            "eligible_positions": ["SP"],
            "games_today": True,
            "accumulated_ip": 5.0,
            "sb": 0.0,
            "hr": 0.0,
        },
        {
            "player_id": "rp1",
            "slot": "RP",
            "eligible_positions": ["RP"],
            "games_today": True,
            "accumulated_ip": 2.0,
            "sb": 0.0,
            "hr": 0.0,
        },
        {
            "player_id": "rp2",
            "slot": "RP",
            "eligible_positions": ["RP"],
            "games_today": True,
            "accumulated_ip": 1.5,
            "sb": 0.0,
            "hr": 0.0,
        },
        {
            "player_id": "p1",
            "slot": "P",
            "eligible_positions": ["SP", "RP", "P"],
            "games_today": True,
            "accumulated_ip": 3.0,
            "sb": 0.0,
            "hr": 0.0,
        },
        {
            "player_id": "p2",
            "slot": "P",
            "eligible_positions": ["SP", "RP", "P"],
            "games_today": True,
            "accumulated_ip": 2.0,
            "sb": 0.0,
            "hr": 0.0,
        },
        # Bench players
        {
            "player_id": "bn1",
            "slot": "BN",
            "eligible_positions": ["OF"],
            "games_today": False if include_no_game_player else True,
            "accumulated_ip": 0.0,
            "sb": 1.0,
            "hr": 0.5,
        },
        {
            "player_id": "bn2",
            "slot": "BN",
            "eligible_positions": ["1B", "OF"],
            "games_today": True,
            "accumulated_ip": 0.0,
            "sb": 0.5,
            "hr": 1.0,
        },
        {
            "player_id": "bn3",
            "slot": "BN",
            "eligible_positions": ["C"],
            "games_today": True,
            "accumulated_ip": 0.0,
            "sb": 0.0,
            "hr": 0.5,
        },
        {
            "player_id": "bn4",
            "slot": "BN",
            "eligible_positions": ["SS", "2B"],
            "games_today": True,
            "accumulated_ip": 0.0,
            "sb": 0.5,
            "hr": 0.0,
        },
        # IL / NA — always ineligible
        {
            "player_id": "il1",
            "slot": "IL",
            "eligible_positions": ["SP"],
            "games_today": False,
            "accumulated_ip": 0.0,
            "sb": 0.0,
            "hr": 0.0,
        },
        {
            "player_id": "il2",
            "slot": "IL",
            "eligible_positions": ["OF"],
            "games_today": False,
            "accumulated_ip": 0.0,
            "sb": 0.0,
            "hr": 0.0,
        },
        {
            "player_id": "na1",
            "slot": "NA",
            "eligible_positions": ["RP"],
            "games_today": False,
            "accumulated_ip": 0.0,
            "sb": 0.0,
            "hr": 0.0,
        },
    ]
    return pd.DataFrame(rows)


def _build_schedule_df(player_ids: list[str]) -> pd.DataFrame:
    """Build a schedule DataFrame with all given players having a game today."""
    return pd.DataFrame(
        [
            {"player_id": pid, "opponent": "OPP", "home_away": "home"}
            for pid in player_ids
        ]
    )


# ---------------------------------------------------------------------------
# 1. optimize_daily_lineup: player without game today NOT in active slot
# ---------------------------------------------------------------------------


def test_optimize_daily_lineup_skips_no_game_players(config: object) -> None:
    from src.config import LeagueSettings

    assert isinstance(config, LeagueSettings)

    roster = _build_roster_df(include_no_game_player=True)
    matchup = _make_matchup_df()

    # Only players with games today (exclude bn1 who has games_today=False)
    players_with_games = cast(
        list[str],
        roster[roster["games_today"] == True]["player_id"].tolist(),  # noqa: E712
    )
    schedule = _build_schedule_df(players_with_games)

    lineup = optimize_daily_lineup(roster, schedule, matchup, config)

    assert "bn1" not in lineup.values(), (
        f"bn1 (no game today) should not be in lineup: {lineup}"
    )


# ---------------------------------------------------------------------------
# 2. optimize_daily_lineup: all 16 active slots filled
# ---------------------------------------------------------------------------


def test_optimize_daily_lineup_fills_all_active_slots(config: object) -> None:
    from src.config import LeagueSettings

    assert isinstance(config, LeagueSettings)

    roster = _build_roster_df()
    matchup = _make_matchup_df()
    # All players have games
    all_player_ids = cast(list[str], roster["player_id"].tolist())
    schedule = _build_schedule_df(all_player_ids)

    lineup = optimize_daily_lineup(roster, schedule, matchup, config)

    # Count expected active slots (16 total)
    assert len(lineup) == 16, (
        f"Expected 16 filled slots, got {len(lineup)}: {list(lineup.keys())}"
    )


# ---------------------------------------------------------------------------
# 3. optimize_daily_lineup: Util slot prefers flippable category player
# ---------------------------------------------------------------------------


def test_optimize_daily_lineup_util_prefers_flippable(config: object) -> None:
    from src.config import LeagueSettings

    assert isinstance(config, LeagueSettings)

    # SB is a flippable category in the matchup
    matchup = _make_matchup_df(
        {
            "h": "safe_win",
            "hr": "safe_win",
            "sb": "flippable_loss",  # SB is contested
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

    roster = _build_roster_df()
    all_player_ids = cast(list[str], roster["player_id"].tolist())
    schedule = _build_schedule_df(all_player_ids)

    lineup = optimize_daily_lineup(roster, schedule, matchup, config)

    # util1 has sb=5.0, which is excellent for the flippable SB category
    # The Util slot should include util1
    util_players = [v for k, v in lineup.items() if "Util" in k]
    assert "util1" in util_players, (
        f"Expected util1 (high SB for flippable category) in a Util slot, "
        f"but Util players are: {util_players}"
    )


# ---------------------------------------------------------------------------
# 4. recommend_adds: respects acquisition limit
# ---------------------------------------------------------------------------


def test_recommend_adds_respects_acquisition_limit(config: object) -> None:
    from src.config import LeagueSettings

    assert isinstance(config, LeagueSettings)

    roster = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "slot": "BN",
                "eligible_positions": ["OF"],
                "overall_score": 1.0,
            }
        ]
    )

    waiver = pd.DataFrame(
        [
            {
                "player_id": f"fa{i}",
                "overall_score": float(10 - i),
                "category_scores": json.dumps({"hr": 1.0, "sb": 0.5}),
                "recommended_drop_id": "p1",
                "is_callup": False,
                "days_since_callup": float("nan"),
            }
            for i in range(5)
        ]
    )

    # 4 used, max=5 → only 1 allowed
    adds = recommend_adds(waiver, roster, acquisitions_used=4, config=config)
    assert len(adds) <= 1, f"Expected at most 1 add, got {len(adds)}"


# ---------------------------------------------------------------------------
# 5. recommend_adds: empty when limit reached
# ---------------------------------------------------------------------------


def test_recommend_adds_empty_when_limit_reached(config: object) -> None:
    from src.config import LeagueSettings

    assert isinstance(config, LeagueSettings)

    roster = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "slot": "BN",
                "eligible_positions": ["OF"],
                "overall_score": 1.0,
            }
        ]
    )
    waiver = pd.DataFrame(
        [
            {
                "player_id": "fa1",
                "overall_score": 9.0,
                "category_scores": json.dumps({"hr": 2.0}),
                "recommended_drop_id": "p1",
                "is_callup": False,
                "days_since_callup": float("nan"),
            }
        ]
    )

    adds = recommend_adds(waiver, roster, acquisitions_used=5, config=config)
    assert adds == [], f"Expected empty list, got {adds}"


# ---------------------------------------------------------------------------
# 6. build_daily_report: output dict has all required keys
# ---------------------------------------------------------------------------


def test_build_daily_report_structure(config: object) -> None:
    from src.config import LeagueSettings

    assert isinstance(config, LeagueSettings)

    lineup = {"C": "c1", "1B": "1b1"}
    adds: list[dict[str, object]] = [
        {
            "add_player_id": "fa1",
            "drop_player_id": "bn1",
            "reason": "Adds value in: hr",
            "score": 3.5,
            "categories_improved": ["hr"],
        }
    ]
    matchup_df = _make_matchup_df()
    ip_pace: dict[str, object] = {
        "current_ip": 15.0,
        "projected_ip": 25.0,
        "min_ip": 21,
        "on_pace": True,
    }
    callup_alerts: list[dict[str, object]] = [
        {
            "player_id": "fa_new",
            "player_name": "John Doe",
            "days_since_callup": 1,
            "team": "NYY",
            "from_level": "AAA",
        }
    ]

    report = build_daily_report(
        lineup=lineup,
        adds=adds,
        matchup_df=matchup_df,
        ip_pace=ip_pace,
        callup_alerts=callup_alerts,
        report_date=datetime.date(2026, 3, 14),
        week_number=1,
    )

    required_keys = {
        "report_date",
        "week_number",
        "lineup",
        "adds",
        "matchup_summary",
        "ip_pace",
        "callup_alerts",
    }
    assert required_keys.issubset(report.keys()), (
        f"Missing keys: {required_keys - set(report.keys())}"
    )
    matchup_summary = report["matchup_summary"]
    assert isinstance(matchup_summary, list), "matchup_summary should be a list"
    assert len(matchup_summary) == len(matchup_df)
    assert report["report_date"] == "2026-03-14"
    assert report["week_number"] == 1


# ---------------------------------------------------------------------------
# 7. build_daily_report: JSON-serializable
# ---------------------------------------------------------------------------


def test_build_daily_report_is_json_serializable(config: object) -> None:
    from src.config import LeagueSettings

    assert isinstance(config, LeagueSettings)

    lineup = {"C": "c1", "1B": "1b1", "SP": "sp1"}
    adds: list[dict[str, object]] = []
    matchup_df = _make_matchup_df()
    ip_pace: dict[str, object] = {
        "current_ip": 12.0,
        "projected_ip": 20.0,
        "min_ip": 21,
        "on_pace": False,
    }
    callup_alerts: list[dict[str, object]] = []

    report = build_daily_report(
        lineup=lineup,
        adds=adds,
        matchup_df=matchup_df,
        ip_pace=ip_pace,
        callup_alerts=callup_alerts,
    )

    # Should not raise
    serialized = json.dumps(report)
    assert isinstance(serialized, str)
    # Round-trip check
    parsed = json.loads(serialized)
    assert parsed["lineup"] == lineup


# ---------------------------------------------------------------------------
# 8. recommend_adds: enriched output fields present
# ---------------------------------------------------------------------------


def _make_enriched_waiver_df() -> pd.DataFrame:
    """Waiver DF with position, streak, callup, and category_scores fields."""
    return pd.DataFrame(
        [
            {
                "player_id": "fa_sp",
                "overall_score": 8.0,
                "category_scores": json.dumps({"k": 2.0, "w": 1.5}),
                "recommended_drop_id": "bench_sp",
                "is_callup": False,
                "days_since_callup": float("nan"),
                "position": "SP",
                "streak": "🔥 Hot",
            }
        ]
    )


def _make_enriched_roster_df() -> pd.DataFrame:
    """Roster DF with position, streak, and overall_score fields."""
    return pd.DataFrame(
        [
            {
                "player_id": "bench_sp",
                "slot": "BN",
                "eligible_positions": ["SP"],
                "overall_score": 2.0,
                "position": "SP",
                "streak": "❄️ Cold",
            }
        ]
    )


def test_recommend_adds_enriched_fields_present(config: object) -> None:
    """Each add dict must contain the new enrichment keys."""
    from src.config import LeagueSettings

    assert isinstance(config, LeagueSettings)

    waiver = _make_enriched_waiver_df()
    roster = _make_enriched_roster_df()
    adds = recommend_adds(waiver, roster, acquisitions_used=0, config=config)

    assert len(adds) == 1
    add = adds[0]
    required_keys = {
        "add_player_id",
        "drop_player_id",
        "score",
        "reason",
        "categories_improved",
        "add_position",
        "add_streak",
        "add_callup_note",
        "drop_position",
        "drop_streak",
        "matchup_context",
    }
    missing = required_keys - set(add.keys())
    assert not missing, f"Missing enrichment keys: {missing}"


def test_recommend_adds_position_populated(config: object) -> None:
    """add_position and drop_position should reflect the position column."""
    from src.config import LeagueSettings

    assert isinstance(config, LeagueSettings)

    waiver = _make_enriched_waiver_df()
    roster = _make_enriched_roster_df()
    adds = recommend_adds(waiver, roster, acquisitions_used=0, config=config)

    assert adds[0]["add_position"] == "SP"
    assert adds[0]["drop_position"] == "SP"


def test_recommend_adds_handles_ndarray_eligible_positions(config: object) -> None:
    """Roster from MotherDuck returns eligible_positions as numpy arrays.

    Regression: _lookup_position used to do `if pos and ...` which blew up on
    ndarray with "truth value of an array is ambiguous". Drop the `position`
    column so the fallback to `eligible_positions` (ndarray) is exercised.
    """
    import numpy as np

    from src.config import LeagueSettings

    assert isinstance(config, LeagueSettings)

    waiver = _make_enriched_waiver_df()
    roster = _make_enriched_roster_df()
    # Force fallback path: drop the explicit position column and swap the
    # eligible_positions list for a numpy array (what DuckDB VARCHAR[] yields).
    roster = roster.drop(columns=["position"])
    roster.at[0, "eligible_positions"] = np.array(["SP"], dtype=object)

    adds = recommend_adds(waiver, roster, acquisitions_used=0, config=config)

    assert adds[0]["drop_position"] == "SP"


def test_recommend_adds_streak_populated(config: object) -> None:
    """add_streak and drop_streak should reflect the streak column."""
    from src.config import LeagueSettings

    assert isinstance(config, LeagueSettings)

    waiver = _make_enriched_waiver_df()
    roster = _make_enriched_roster_df()
    adds = recommend_adds(waiver, roster, acquisitions_used=0, config=config)

    assert adds[0]["add_streak"] == "🔥 Hot"
    assert adds[0]["drop_streak"] == "❄️ Cold"


def test_recommend_adds_callup_note_for_recent_callup(config: object) -> None:
    """add_callup_note should be set when is_callup=True and days_since_callup is populated."""
    from src.config import LeagueSettings

    assert isinstance(config, LeagueSettings)

    waiver = pd.DataFrame(
        [
            {
                "player_id": "fa_callup",
                "overall_score": 7.0,
                "category_scores": json.dumps({"sb": 2.0}),
                "recommended_drop_id": "bench_of",
                "is_callup": True,
                "days_since_callup": 3.0,
                "position": "OF",
                "streak": "—",
            }
        ]
    )
    roster = pd.DataFrame(
        [
            {
                "player_id": "bench_of",
                "slot": "BN",
                "eligible_positions": ["OF"],
                "overall_score": 1.5,
                "position": "OF",
                "streak": "—",
            }
        ]
    )
    adds = recommend_adds(waiver, roster, acquisitions_used=0, config=config)
    assert len(adds) == 1
    assert "3" in str(adds[0]["add_callup_note"]), (
        f"Expected days_since_callup in callup note, got: {adds[0]['add_callup_note']}"
    )


def test_recommend_adds_matchup_context_with_matchup_df(config: object) -> None:
    """matchup_context should be non-empty when a matchup_df is supplied."""
    from src.config import LeagueSettings

    assert isinstance(config, LeagueSettings)

    waiver = _make_enriched_waiver_df()
    roster = _make_enriched_roster_df()
    matchup = _make_matchup_df({"k": "flippable_loss", "w": "toss_up"})
    adds = recommend_adds(
        waiver, roster, acquisitions_used=0, config=config, matchup_df=matchup
    )
    assert len(adds) == 1
    assert adds[0]["matchup_context"] != "", (
        "matchup_context should be populated when matchup_df is provided"
    )
