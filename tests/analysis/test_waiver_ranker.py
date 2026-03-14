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
