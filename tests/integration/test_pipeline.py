"""
tests/integration/test_pipeline.py

Integration tests for the full daily pipeline.

Uses in-memory DuckDB with pre-seeded data. All external API calls
(Yahoo, MLB Stats API, pybaseball) are mocked.

These tests verify that the complete data flow from API → DB → analysis
→ daily report works end-to-end with realistic data.
"""

from __future__ import annotations

import datetime
import json
import unittest.mock as mock

import duckdb
import pandas as pd
import pytest

from src.config import load_league_settings
from src.db.loaders_mlb import get_fantasy_week
from src.db.schema import (
    DIM_PLAYERS,
    FACT_DAILY_REPORTS,
    FACT_PIPELINE_RUNS,
    FACT_PLAYER_STATS_DAILY,
    FACT_ROSTERS,
    FACT_WAIVER_SCORES,
    create_all_tables,
)
from src.pipeline.daily_run import (
    _get_season_start,
    _step_write_daily_report,
    run_daily_pipeline,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def conn() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB connection with all tables pre-created."""
    c = duckdb.connect(":memory:")
    create_all_tables(c)
    yield c
    c.close()


@pytest.fixture()
def settings():
    return load_league_settings()


@pytest.fixture()
def today() -> datetime.date:
    return datetime.date(2026, 4, 8)  # A Wednesday in week 2


@pytest.fixture()
def week(today) -> int:
    return get_fantasy_week(today, _get_season_start(today.year))


@pytest.fixture()
def seeded_conn(conn, today, week) -> duckdb.DuckDBPyConnection:
    """Connection pre-seeded with realistic fixture data.

    Inserts:
    - dim_players: 3 batters + 2 pitchers on my team, 3 FA
    - fact_rosters: my team's current roster snapshot
    - fact_player_stats_daily: 2 days of stats for this week
    - fact_waiver_scores: staged FA records (score=0)
    """
    my_team_key = "423.l.87941.t.10"
    week_start = today - datetime.timedelta(days=today.weekday())

    # ── dim_players ────────────────────────────────────────────────────────────
    players = pd.DataFrame(
        [
            # My roster
            {
                "player_id": "423.p.1",
                "full_name": "Aaron Judge",
                "mlb_id": 592450,
                "fg_id": "16378",
                "team": "NYY",
                "positions": ["OF"],
                "bats": "R",
                "throws": "R",
                "status": "Active",
                "updated_at": None,
            },
            {
                "player_id": "423.p.2",
                "full_name": "Freddie Freeman",
                "mlb_id": 518692,
                "fg_id": "7163",
                "team": "LAD",
                "positions": ["1B"],
                "bats": "L",
                "throws": "R",
                "status": "Active",
                "updated_at": None,
            },
            {
                "player_id": "423.p.3",
                "full_name": "Gerrit Cole",
                "mlb_id": 543037,
                "fg_id": "13125",
                "team": "NYY",
                "positions": ["SP"],
                "bats": "R",
                "throws": "R",
                "status": "Active",
                "updated_at": None,
            },
            {
                "player_id": "423.p.4",
                "full_name": "Emmanuel Clase",
                "mlb_id": 669373,
                "fg_id": "22235",
                "team": "CLE",
                "positions": ["RP"],
                "bats": "R",
                "throws": "R",
                "status": "Active",
                "updated_at": None,
            },
            {
                "player_id": "423.p.5",
                "full_name": "Yordan Alvarez",
                "mlb_id": 670541,
                "fg_id": "19556",
                "team": "HOU",
                "positions": ["OF", "Util"],
                "bats": "L",
                "throws": "R",
                "status": "Active",
                "updated_at": None,
            },
            # Free agents
            {
                "player_id": "423.p.6",
                "full_name": "Jackson Chourio",
                "mlb_id": 672921,
                "fg_id": "26490",
                "team": "MIL",
                "positions": ["OF"],
                "bats": "R",
                "throws": "R",
                "status": "Active",
                "updated_at": None,
            },
            {
                "player_id": "423.p.7",
                "full_name": "Kyle Freeland",
                "mlb_id": 621433,
                "fg_id": "15678",
                "team": "COL",
                "positions": ["SP"],
                "bats": "L",
                "throws": "L",
                "status": "Active",
                "updated_at": None,
            },
        ]
    )
    conn.register("_players", players)
    conn.execute(f"""
        INSERT OR REPLACE INTO {DIM_PLAYERS}
            (player_id, mlb_id, fg_id, full_name, team, positions, bats, throws, status, updated_at)
        SELECT player_id, mlb_id, fg_id, full_name, team, positions, bats, throws, status, updated_at
        FROM _players
    """)
    conn.unregister("_players")

    # ── fact_rosters ───────────────────────────────────────────────────────────
    rosters = pd.DataFrame(
        [
            {
                "team_id": my_team_key,
                "player_id": "423.p.1",
                "snapshot_date": today,
                "roster_slot": "OF",
                "acquisition_type": "draft",
            },
            {
                "team_id": my_team_key,
                "player_id": "423.p.2",
                "snapshot_date": today,
                "roster_slot": "1B",
                "acquisition_type": "draft",
            },
            {
                "team_id": my_team_key,
                "player_id": "423.p.3",
                "snapshot_date": today,
                "roster_slot": "SP",
                "acquisition_type": "draft",
            },
            {
                "team_id": my_team_key,
                "player_id": "423.p.4",
                "snapshot_date": today,
                "roster_slot": "RP",
                "acquisition_type": "waiver",
            },
            {
                "team_id": my_team_key,
                "player_id": "423.p.5",
                "snapshot_date": today,
                "roster_slot": "Util",
                "acquisition_type": "draft",
            },
        ]
    )
    conn.register("_rosters", rosters)
    conn.execute(f"INSERT OR REPLACE INTO {FACT_ROSTERS} SELECT * FROM _rosters")
    conn.unregister("_rosters")

    # ── fact_player_stats_daily (2 days of stats) ─────────────────────────────
    stats_rows = []
    for day_offset in range(2):
        stat_date = week_start + datetime.timedelta(days=day_offset)
        stats_rows.extend(
            [
                {
                    "player_id": "423.p.1",
                    "stat_date": stat_date,
                    "ab": 4,
                    "h": 2,
                    "hr": 1,
                    "sb": 0,
                    "bb": 1,
                    "hbp": 0,
                    "sf": 0,
                    "tb": 5,
                    "errors": 0,
                    "chances": 3,
                    "ip": None,
                    "w": None,
                    "k": None,
                    "walks_allowed": None,
                    "hits_allowed": None,
                    "sv": None,
                    "holds": None,
                    "avg": 0.500,
                    "ops": 0.950,
                    "fpct": 1.0,
                    "whip": None,
                    "k_bb": None,
                    "sv_h": None,
                },
                {
                    "player_id": "423.p.3",
                    "stat_date": stat_date,
                    "ab": None,
                    "h": None,
                    "hr": None,
                    "sb": None,
                    "bb": None,
                    "hbp": None,
                    "sf": None,
                    "tb": None,
                    "errors": 0,
                    "chances": 2,
                    "ip": 6.0,
                    "w": 1,
                    "k": 8,
                    "walks_allowed": 2,
                    "hits_allowed": 5,
                    "sv": 0,
                    "holds": 0,
                    "avg": None,
                    "ops": None,
                    "fpct": 1.0,
                    "whip": 1.167,
                    "k_bb": 4.0,
                    "sv_h": 0,
                },
            ]
        )
    stats_df = pd.DataFrame(stats_rows)
    conn.register("_stats", stats_df)
    conn.execute(
        f"INSERT OR REPLACE INTO {FACT_PLAYER_STATS_DAILY} SELECT * FROM _stats"
    )
    conn.unregister("_stats")

    # ── fact_waiver_scores (staged with score=0) ───────────────────────────────
    fa_staged = pd.DataFrame(
        [
            {
                "player_id": "423.p.6",
                "score_date": today,
                "overall_score": 0.0,
                "category_scores": None,
                "is_callup": False,
                "days_since_callup": None,
                "recommended_drop_id": None,
                "notes": None,
            },
            {
                "player_id": "423.p.7",
                "score_date": today,
                "overall_score": 0.0,
                "category_scores": None,
                "is_callup": False,
                "days_since_callup": None,
                "recommended_drop_id": None,
                "notes": None,
            },
        ]
    )
    conn.register("_fa", fa_staged)
    conn.execute(f"INSERT OR REPLACE INTO {FACT_WAIVER_SCORES} SELECT * FROM _fa")
    conn.unregister("_fa")

    return conn


# ── Integration: _step_write_daily_report → query ──────────────────────────────


def test_daily_report_roundtrip(seeded_conn, today, week):
    """Write a daily report and read it back — JSON structure is preserved."""
    report = {
        "report_date": today.isoformat(),
        "week_number": week,
        "lineup": {"OF": "423.p.1", "1B": "423.p.2", "SP": "423.p.3"},
        "adds": [
            {
                "add_player_id": "423.p.6",
                "drop_player_id": "423.p.4",
                "reason": "Adds value in: h, hr",
                "score": 12.5,
                "categories_improved": ["h", "hr"],
            }
        ],
        "matchup_summary": [
            {
                "category": "h",
                "my_value": 10.0,
                "opp_value": 8.0,
                "my_leads": True,
                "margin_pct": 0.2,
                "win_prob": 0.7,
                "status": "flippable_win",
            },
        ],
        "ip_pace": {
            "current_ip": 12.0,
            "projected_ip": 28.0,
            "min_ip": 21,
            "on_pace": True,
        },
        "callup_alerts": [],
    }

    _step_write_daily_report(
        seeded_conn, report, today, week, today.year, "run-integ-1"
    )

    row = seeded_conn.execute(
        f"SELECT report_json, week_number, season FROM {FACT_DAILY_REPORTS} WHERE report_date = ?",
        [today],
    ).fetchone()
    assert row is not None
    parsed = json.loads(row[0])

    assert parsed["week_number"] == week
    assert parsed["lineup"]["OF"] == "423.p.1"
    assert len(parsed["adds"]) == 1
    assert parsed["adds"][0]["add_player_id"] == "423.p.6"
    assert parsed["ip_pace"]["on_pace"] is True
    assert row[2] == today.year


# ── Integration: full pipeline with seeded data and mocked APIs ────────────────


def _make_mock_roster_df(team_key: str, today: datetime.date) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team_id": team_key,
                "player_id": "423.p.1",
                "snapshot_date": today,
                "roster_slot": "OF",
                "acquisition_type": "draft",
            },
            {
                "team_id": team_key,
                "player_id": "423.p.3",
                "snapshot_date": today,
                "roster_slot": "SP",
                "acquisition_type": "draft",
            },
        ]
    )


def _make_mock_all_rosters_df(today: datetime.date) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "team_id",
            "player_id",
            "snapshot_date",
            "roster_slot",
            "acquisition_type",
        ]
    )


def _make_mock_transactions_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "transaction_id",
            "league_id",
            "transaction_date",
            "type",
            "team_id",
            "player_id",
            "from_team_id",
            "notes",
        ]
    )


def _make_mock_players_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": "423.p.1",
                "full_name": "Aaron Judge",
                "mlb_id": 592450,
                "fg_id": "16378",
                "team": "NYY",
                "positions": ["OF"],
                "bats": "R",
                "throws": "R",
                "status": "Active",
                "updated_at": None,
            },
        ]
    )


def _make_mock_fa_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": "423.p.6",
                "full_name": "Jackson Chourio",
                "team": "MIL",
                "positions": ["OF"],
                "h": 1.5,
                "hr": 0.3,
                "sb": 0.2,
                "bb": 0.4,
                "ab": 4.0,
                "avg": 0.300,
                "ops": 0.850,
                "w": 0,
                "k": 0,
                "whip": 0.0,
                "k_bb": 0.0,
                "sv_h": 0,
                "fpct": 0.990,
            },
        ]
    )


def test_full_pipeline_with_mocked_apis(conn, settings, today, monkeypatch):
    """End-to-end pipeline: mocked APIs + in-memory DB → report written."""
    team_key = "423.l.87941.t.10"
    monkeypatch.setenv("YAHOO_TEAM_ID", "10")

    mock_yahoo = mock.MagicMock()
    mock_yahoo.get_my_roster.return_value = _make_mock_roster_df(team_key, today)
    mock_yahoo.get_all_rosters.return_value = _make_mock_all_rosters_df(today)
    mock_yahoo.get_transactions.return_value = _make_mock_transactions_df()
    mock_yahoo.get_player_details.return_value = _make_mock_players_df()
    mock_yahoo.get_free_agents.return_value = _make_mock_fa_df()

    empty_df = pd.DataFrame()

    with (
        mock.patch(
            "src.pipeline.daily_run.YahooClient.from_env", return_value=mock_yahoo
        ),
        mock.patch(
            "src.pipeline.daily_run.mlb_client.get_batter_stats", return_value=empty_df
        ),
        mock.patch(
            "src.pipeline.daily_run.mlb_client.get_pitcher_stats", return_value=empty_df
        ),
        mock.patch(
            "src.pipeline.daily_run.mlb_client.get_daily_game_schedule",
            return_value=empty_df,
        ),
        mock.patch(
            "src.pipeline.daily_run.mlb_client.get_steamer_projections",
            return_value=empty_df,
        ),
        mock.patch(
            "src.pipeline.daily_run.mlb_client.get_recent_callups",
            return_value=empty_df,
        ),
    ):
        result = run_daily_pipeline(conn, settings, run_date=today)

    assert "run_id" in result
    assert result["status"] in ("success", "partial")

    # Pipeline run logged
    run_row = conn.execute(
        f"SELECT status FROM {FACT_PIPELINE_RUNS} WHERE run_id = ?",
        [result["run_id"]],
    ).fetchone()
    assert run_row is not None

    # Rosters were loaded (Yahoo mock returned data)
    roster_count = conn.execute(f"SELECT COUNT(*) FROM {FACT_ROSTERS}").fetchone()[0]
    assert roster_count > 0

    # Daily report written (analysis ran with empty stats → valid but zero-filled report)
    report_row = conn.execute(
        f"SELECT report_json FROM {FACT_DAILY_REPORTS} WHERE report_date = ?",
        [today],
    ).fetchone()
    assert report_row is not None
    report_data = json.loads(report_row[0])
    assert "week_number" in report_data
    assert "lineup" in report_data
    assert "ip_pace" in report_data


def test_pipeline_survives_yahoo_failure(conn, settings, today, monkeypatch):
    """Pipeline reaches partial/failed status when Yahoo is unavailable."""
    empty_df = pd.DataFrame()

    with (
        mock.patch(
            "src.pipeline.daily_run.YahooClient.from_env",
            side_effect=Exception("OAuth failed"),
        ),
        mock.patch(
            "src.pipeline.daily_run.mlb_client.get_batter_stats", return_value=empty_df
        ),
        mock.patch(
            "src.pipeline.daily_run.mlb_client.get_pitcher_stats", return_value=empty_df
        ),
        mock.patch(
            "src.pipeline.daily_run.mlb_client.get_daily_game_schedule",
            return_value=empty_df,
        ),
        mock.patch(
            "src.pipeline.daily_run.mlb_client.get_steamer_projections",
            return_value=empty_df,
        ),
        mock.patch(
            "src.pipeline.daily_run.mlb_client.get_recent_callups",
            return_value=empty_df,
        ),
    ):
        result = run_daily_pipeline(conn, settings, run_date=today)

    # Pipeline should not raise — status reflects partial/failed
    assert result["status"] in ("partial", "failed")
    # But still logged the run
    run_row = conn.execute(
        f"SELECT status FROM {FACT_PIPELINE_RUNS} WHERE run_id = ?",
        [result["run_id"]],
    ).fetchone()
    assert run_row is not None


def test_pipeline_idempotent(conn, settings, today, monkeypatch):
    """Running the pipeline twice on the same day replaces the report (idempotent)."""
    monkeypatch.setenv("YAHOO_TEAM_ID", "10")
    team_key = "423.l.87941.t.10"

    mock_yahoo = mock.MagicMock()
    mock_yahoo.get_my_roster.return_value = _make_mock_roster_df(team_key, today)
    mock_yahoo.get_all_rosters.return_value = _make_mock_all_rosters_df(today)
    mock_yahoo.get_transactions.return_value = _make_mock_transactions_df()
    mock_yahoo.get_player_details.return_value = _make_mock_players_df()
    mock_yahoo.get_free_agents.return_value = _make_mock_fa_df()

    empty_df = pd.DataFrame()

    with (
        mock.patch(
            "src.pipeline.daily_run.YahooClient.from_env", return_value=mock_yahoo
        ),
        mock.patch(
            "src.pipeline.daily_run.mlb_client.get_batter_stats", return_value=empty_df
        ),
        mock.patch(
            "src.pipeline.daily_run.mlb_client.get_pitcher_stats", return_value=empty_df
        ),
        mock.patch(
            "src.pipeline.daily_run.mlb_client.get_daily_game_schedule",
            return_value=empty_df,
        ),
        mock.patch(
            "src.pipeline.daily_run.mlb_client.get_steamer_projections",
            return_value=empty_df,
        ),
        mock.patch(
            "src.pipeline.daily_run.mlb_client.get_recent_callups",
            return_value=empty_df,
        ),
    ):
        run_daily_pipeline(conn, settings, run_date=today)
        run_daily_pipeline(conn, settings, run_date=today)

    # Only one report per day (upsert)
    report_count = conn.execute(
        f"SELECT COUNT(*) FROM {FACT_DAILY_REPORTS}"
    ).fetchone()[0]
    assert report_count == 1

    # Two distinct pipeline run records
    run_count = conn.execute(f"SELECT COUNT(*) FROM {FACT_PIPELINE_RUNS}").fetchone()[0]
    assert run_count == 2
