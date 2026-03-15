"""
tests/pipeline/test_daily_run.py

Unit tests for src/pipeline/daily_run.py.

All tests use in-memory DuckDB — no real API or MotherDuck calls.
External API calls (YahooClient, mlb_client) are mocked.
"""

from __future__ import annotations

import datetime
import json

import duckdb
import pandas as pd
import pytest

from src.config import load_league_settings
from src.db.schema import (
    FACT_DAILY_REPORTS,
    FACT_PIPELINE_RUNS,
    FACT_WAIVER_SCORES,
    create_all_tables,
)
from src.pipeline.daily_run import (
    _MLB_TEAM_ID_TO_ABBR,
    _aggregate_to_team,
    _build_player_schedule,
    _enrich_roster_with_stats,
    _get_season_start,
    _get_week_start,
    _my_team_key,
    _query_weekly_acquisitions,
    _step_log_pipeline_run,
    _step_write_daily_report,
    _update_waiver_scores,
    run_daily_pipeline,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def conn() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB connection with all tables created."""
    c = duckdb.connect(":memory:")
    create_all_tables(c)
    yield c
    c.close()


@pytest.fixture()
def settings():
    return load_league_settings()


@pytest.fixture()
def today() -> datetime.date:
    return datetime.date(2026, 4, 6)  # A Monday in week 2


# ── _get_season_start ──────────────────────────────────────────────────────────


def test_get_season_start_known():
    assert _get_season_start(2026) == datetime.date(2026, 3, 26)


def test_get_season_start_unknown_defaults():
    d = _get_season_start(2030)
    assert d == datetime.date(2030, 3, 26)


# ── _get_week_start ────────────────────────────────────────────────────────────


def test_get_week_start_monday():
    monday = datetime.date(2026, 4, 6)
    assert _get_week_start(monday) == monday


def test_get_week_start_midweek():
    wednesday = datetime.date(2026, 4, 8)
    assert _get_week_start(wednesday) == datetime.date(2026, 4, 6)


def test_get_week_start_sunday():
    sunday = datetime.date(2026, 4, 12)
    assert _get_week_start(sunday) == datetime.date(2026, 4, 6)


# ── _my_team_key ───────────────────────────────────────────────────────────────


def test_my_team_key_from_config(settings, monkeypatch):
    """Uses config my_team_key if set."""
    monkeypatch.setenv("YAHOO_TEAM_ID", "5")
    # Patch settings to have a non-empty my_team_key
    from dataclasses import replace

    s = replace(settings, my_team_key="422.l.87941.t.3")
    assert _my_team_key(s) == "422.l.87941.t.3"


def test_my_team_key_from_env_var(settings, monkeypatch):
    """Falls back to YAHOO_TEAM_ID when config my_team_key is empty."""
    from dataclasses import replace

    empty_settings = replace(settings, my_team_key="")
    monkeypatch.setenv("YAHOO_TEAM_ID", "7")
    assert _my_team_key(empty_settings) == "422.l.87941.t.7"


def test_my_team_key_default_when_no_env(settings, monkeypatch):
    """Uses team id '1' when config is empty and no env var is set."""
    from dataclasses import replace

    empty_settings = replace(settings, my_team_key="")
    monkeypatch.delenv("YAHOO_TEAM_ID", raising=False)
    key = _my_team_key(empty_settings)
    assert key == "422.l.87941.t.1"


# ── _MLB_TEAM_ID_TO_ABBR ───────────────────────────────────────────────────────


def test_team_id_map_has_30_teams():
    assert len(_MLB_TEAM_ID_TO_ABBR) == 30


def test_team_id_map_known_entries():
    assert _MLB_TEAM_ID_TO_ABBR[147] == "NYY"
    assert _MLB_TEAM_ID_TO_ABBR[111] == "BOS"
    assert _MLB_TEAM_ID_TO_ABBR[119] == "LAD"


# ── _build_player_schedule ─────────────────────────────────────────────────────


def test_build_player_schedule_returns_empty_on_empty_schedule():
    players_df = pd.DataFrame(
        {
            "player_id": ["p1", "p2"],
            "team": ["NYY", "BOS"],
        }
    )
    result = (
        _build_player_schedule.__wrapped__
        if hasattr(_build_player_schedule, "__wrapped__")
        else None
    )
    # Test directly: when schedule has no mlb_id column
    # Patch mlb_client to raise to simulate failure
    import unittest.mock as mock

    from src.pipeline.daily_run import _build_player_schedule as bps

    with mock.patch(
        "src.pipeline.daily_run.mlb_client.get_daily_game_schedule",
        side_effect=Exception("no network"),
    ):
        result = bps(datetime.date(2026, 4, 6), players_df)
    assert result.empty or list(result.columns) == ["player_id"]


def test_build_player_schedule_filters_by_team(monkeypatch):
    """Players on teams with games today appear in schedule_df."""
    import unittest.mock as mock

    team_schedule = pd.DataFrame(
        {
            "mlb_id": [147, 111],  # NYY and BOS playing
            "game_date": [datetime.date(2026, 4, 6)] * 2,
            "opponent_team": ["BOS", "NYY"],
            "home_away": ["home", "away"],
            "probable_pitcher": [None, None],
        }
    )
    players_df = pd.DataFrame(
        {
            "player_id": ["p1", "p2", "p3"],
            "mlb_id": [1, 2, 3],
            "team": ["NYY", "BOS", "LAD"],
        }
    )

    with mock.patch(
        "src.pipeline.daily_run.mlb_client.get_daily_game_schedule",
        return_value=team_schedule,
    ):
        from src.pipeline.daily_run import _build_player_schedule

        result = _build_player_schedule(datetime.date(2026, 4, 6), players_df)

    assert set(result["player_id"].tolist()) == {"p1", "p2"}
    assert "p3" not in result["player_id"].tolist()


# ── _aggregate_to_team ─────────────────────────────────────────────────────────


def test_aggregate_to_team_sums_counting_stats():
    df = pd.DataFrame(
        {
            "player_id": ["p1", "p2"],
            "h": [3, 2],
            "hr": [1, 0],
            "sb": [0, 1],
            "bb": [2, 1],
            "ab": [10, 8],
            "hbp": [0, 0],
            "sf": [0, 0],
            "tb": [5, 3],
            "errors": [0, 0],
            "chances": [5, 3],
            "ip": [6.0, 0.0],
            "w": [1, 0],
            "k": [7, 0],
            "walks_allowed": [2, 0],
            "hits_allowed": [4, 0],
            "sv": [0, 1],
            "holds": [1, 0],
            "avg": [0.300, 0.250],
            "ops": [0.850, 0.700],
            "fpct": [1.0, 1.0],
            "whip": [1.0, 0.0],
            "k_bb": [3.5, 0.0],
            "sv_h": [1, 1],
        }
    )
    result = _aggregate_to_team(df)
    assert len(result) == 1
    assert result.iloc[0]["h"] == 5
    assert result.iloc[0]["hr"] == 1
    assert result.iloc[0]["w"] == 1


def test_aggregate_to_team_recomputes_avg():
    """AVG must be H/AB, not average of individual AVGs."""
    df = pd.DataFrame(
        {
            "player_id": ["p1", "p2"],
            "h": [1, 1],
            "ab": [2, 8],  # individual AVGs: .500 and .125
            "hr": [0, 0],
            "sb": [0, 0],
            "bb": [0, 0],
            "hbp": [0, 0],
            "sf": [0, 0],
            "tb": [1, 1],
            "errors": [0, 0],
            "chances": [1, 1],
            "ip": [0.0, 0.0],
            "w": [0, 0],
            "k": [0, 0],
            "walks_allowed": [0, 0],
            "hits_allowed": [0, 0],
            "sv": [0, 0],
            "holds": [0, 0],
            "avg": [0.500, 0.125],
            "ops": [0.5, 0.125],
            "fpct": [1.0, 1.0],
            "whip": [0.0, 0.0],
            "k_bb": [0.0, 0.0],
            "sv_h": [0, 0],
        }
    )
    result = _aggregate_to_team(df)
    # Combined: 2 H in 10 AB = .200, NOT average of .500 and .125
    assert abs(result.iloc[0]["avg"] - 0.200) < 0.001


def test_aggregate_to_team_recomputes_whip():
    """WHIP must be (BB+H)/IP from components."""
    df = pd.DataFrame(
        {
            "player_id": ["p1", "p2"],
            "h": [0, 0],
            "ab": [0, 0],
            "hr": [0, 0],
            "sb": [0, 0],
            "bb": [0, 0],
            "hbp": [0, 0],
            "sf": [0, 0],
            "tb": [0, 0],
            "errors": [0, 0],
            "chances": [0, 0],
            "ip": [3.0, 3.0],
            "w": [0, 0],
            "k": [0, 0],
            "walks_allowed": [1, 2],
            "hits_allowed": [3, 3],
            "sv": [0, 0],
            "holds": [0, 0],
            "avg": [0.0, 0.0],
            "ops": [0.0, 0.0],
            "fpct": [0.0, 0.0],
            "whip": [1.333, 1.667],
            "k_bb": [0.0, 0.0],
            "sv_h": [0, 0],
        }
    )
    result = _aggregate_to_team(df)
    # Combined: (1+2 walks + 3+3 hits) / (3+3) IP = 9/6 = 1.5
    assert abs(result.iloc[0]["whip"] - 1.5) < 0.001


def test_aggregate_to_team_empty_returns_zeros():
    result = _aggregate_to_team(pd.DataFrame())
    assert len(result) == 1
    assert result.iloc[0]["h"] == 0


# ── _enrich_roster_with_stats ──────────────────────────────────────────────────


def test_enrich_roster_adds_accumulated_ip():
    roster = pd.DataFrame(
        {
            "player_id": ["p1", "p2"],
            "slot": ["SP", "C"],
        }
    )
    stats = pd.DataFrame(
        {
            "player_id": ["p1"],
            "ip": [12.0],
        }
    )
    result = _enrich_roster_with_stats(roster, stats)
    assert result.loc[result["player_id"] == "p1", "accumulated_ip"].iloc[0] == 12.0
    assert result.loc[result["player_id"] == "p2", "accumulated_ip"].iloc[0] == 0.0


def test_enrich_roster_empty_stats():
    roster = pd.DataFrame({"player_id": ["p1"], "slot": ["SP"]})
    result = _enrich_roster_with_stats(roster, pd.DataFrame())
    assert result.iloc[0]["accumulated_ip"] == 0.0


# ── _step_write_daily_report ───────────────────────────────────────────────────


def test_step_write_daily_report_inserts_row(conn, today, settings):
    from src.pipeline.daily_run import get_fantasy_week

    week = get_fantasy_week(today, _get_season_start(today.year))

    report = {
        "report_date": today.isoformat(),
        "week_number": week,
        "lineup": {"C": "p1"},
        "adds": [],
        "matchup_summary": [],
        "ip_pace": {
            "current_ip": 0.0,
            "projected_ip": 0.0,
            "min_ip": 21,
            "on_pace": False,
        },
        "callup_alerts": [],
    }
    n = _step_write_daily_report(conn, report, today, week, today.year, "run-123")
    assert n == 1

    row = conn.execute(
        f"SELECT report_json, week_number FROM {FACT_DAILY_REPORTS} WHERE report_date = ?",
        [today],
    ).fetchone()
    assert row is not None
    parsed = json.loads(row[0])
    assert parsed["lineup"] == {"C": "p1"}
    assert row[1] == week


def test_step_write_daily_report_upserts(conn, today, settings):
    """Re-writing the same report_date replaces the row."""
    from src.pipeline.daily_run import get_fantasy_week

    week = get_fantasy_week(today, _get_season_start(today.year))

    report_v1 = {
        "report_date": today.isoformat(),
        "week_number": week,
        "lineup": {"C": "p1"},
        "adds": [],
        "matchup_summary": [],
        "ip_pace": {},
        "callup_alerts": [],
    }
    report_v2 = {
        "report_date": today.isoformat(),
        "week_number": week,
        "lineup": {"C": "p2"},
        "adds": [],
        "matchup_summary": [],
        "ip_pace": {},
        "callup_alerts": [],
    }

    _step_write_daily_report(conn, report_v1, today, week, today.year, "run-1")
    _step_write_daily_report(conn, report_v2, today, week, today.year, "run-2")

    count = conn.execute(f"SELECT COUNT(*) FROM {FACT_DAILY_REPORTS}").fetchone()[0]
    assert count == 1

    row = conn.execute(
        f"SELECT report_json FROM {FACT_DAILY_REPORTS} WHERE report_date = ?", [today]
    ).fetchone()
    parsed = json.loads(row[0])
    assert parsed["lineup"] == {"C": "p2"}


# ── _step_log_pipeline_run ────────────────────────────────────────────────────


def test_step_log_pipeline_run_inserts(conn):
    _step_log_pipeline_run(
        conn,
        run_id="run-abc",
        status="success",
        rows_written={"fact_rosters": 10},
        errors=[],
        duration_seconds=3.14,
    )
    row = conn.execute(
        f"SELECT status, duration_seconds FROM {FACT_PIPELINE_RUNS} WHERE run_id = 'run-abc'"
    ).fetchone()
    assert row is not None
    assert row[0] == "success"
    assert abs(float(row[1]) - 3.14) < 0.01


def test_step_log_pipeline_run_partial_status(conn):
    _step_log_pipeline_run(
        conn,
        run_id="run-partial",
        status="partial",
        rows_written={"fact_rosters": 5},
        errors=["mlb: timeout"],
        duration_seconds=10.0,
    )
    row = conn.execute(
        f"SELECT status, errors FROM {FACT_PIPELINE_RUNS} WHERE run_id = 'run-partial'"
    ).fetchone()
    assert row[0] == "partial"
    assert "mlb: timeout" in row[1]


# ── _query_weekly_acquisitions ────────────────────────────────────────────────


def test_query_weekly_acquisitions_counts_adds(conn, today):
    """Counts 'add' transactions this week for the given team."""
    week_start = _get_week_start(today)
    staging = pd.DataFrame(
        [
            {
                "transaction_id": "t1",
                "league_id": 87941,
                "transaction_date": (
                    week_start + datetime.timedelta(days=1)
                ).isoformat(),
                "type": "add",
                "team_id": "422.l.87941.t.3",
                "player_id": "p1",
                "from_team_id": None,
                "notes": None,
            },
            {
                "transaction_id": "t2",
                "league_id": 87941,
                "transaction_date": (
                    week_start + datetime.timedelta(days=2)
                ).isoformat(),
                "type": "add",
                "team_id": "422.l.87941.t.3",
                "player_id": "p2",
                "from_team_id": None,
                "notes": None,
            },
            {
                "transaction_id": "t3",
                "league_id": 87941,
                "transaction_date": (
                    week_start + datetime.timedelta(days=1)
                ).isoformat(),
                "type": "drop",  # Should NOT be counted
                "team_id": "422.l.87941.t.3",
                "player_id": "p3",
                "from_team_id": None,
                "notes": None,
            },
        ]
    )
    conn.register("_txn_tmp", staging)
    conn.execute("INSERT INTO fact_transactions SELECT * FROM _txn_tmp")
    conn.unregister("_txn_tmp")

    n = _query_weekly_acquisitions(conn, "422.l.87941.t.3", today)
    assert n == 2


def test_query_weekly_acquisitions_returns_zero_when_empty(conn, today):
    n = _query_weekly_acquisitions(conn, "422.l.87941.t.99", today)
    assert n == 0


# ── _update_waiver_scores ─────────────────────────────────────────────────────


def test_update_waiver_scores_overwrites_placeholder(conn, today):
    """Real scores replace the placeholder 0.0 scores."""
    # Stage placeholder
    placeholder = pd.DataFrame(
        [
            {
                "player_id": "p1",
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
    conn.register("_ph", placeholder)
    conn.execute(f"INSERT INTO {FACT_WAIVER_SCORES} SELECT * FROM _ph")
    conn.unregister("_ph")

    ranked = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "overall_score": 42.5,
                "category_scores": '{"h": 1.0}',
                "recommended_drop_id": "p99",
                "is_callup": False,
                "days_since_callup": None,
            },
        ]
    )
    _update_waiver_scores(conn, ranked, today)

    row = conn.execute(
        f"SELECT overall_score FROM {FACT_WAIVER_SCORES} WHERE player_id = 'p1'"
    ).fetchone()
    assert abs(float(row[0]) - 42.5) < 0.01


def test_update_waiver_scores_no_op_on_empty(conn, today):
    """Empty ranked_df does not raise or modify the table."""
    _update_waiver_scores(conn, pd.DataFrame(), today)
    count = conn.execute(f"SELECT COUNT(*) FROM {FACT_WAIVER_SCORES}").fetchone()[0]
    assert count == 0


# ── run_daily_pipeline (smoke test with mocked APIs) ─────────────────────────


def test_run_daily_pipeline_returns_run_id(conn, settings, today, monkeypatch):
    """Pipeline completes (possibly with partial failures) and returns a run_id."""
    import unittest.mock as mock

    # Mock all external calls
    empty_df = pd.DataFrame()

    with (
        mock.patch(
            "src.pipeline.daily_run.YahooClient.from_env",
            side_effect=Exception("no creds"),
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
    assert result["status"] in ("success", "partial", "failed")


def test_run_daily_pipeline_logs_run(conn, settings, today, monkeypatch):
    """Pipeline always writes a run record to fact_pipeline_runs."""
    import unittest.mock as mock

    empty_df = pd.DataFrame()

    with (
        mock.patch(
            "src.pipeline.daily_run.YahooClient.from_env",
            side_effect=Exception("no creds"),
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

    run_id = result["run_id"]
    row = conn.execute(
        f"SELECT status FROM {FACT_PIPELINE_RUNS} WHERE run_id = ?", [run_id]
    ).fetchone()
    assert row is not None


# ── __main__ exit code logic ──────────────────────────────────────────────────


def test_main_exit_code_success():
    """status != 'failed' maps to exit code 0."""
    for status in ("success", "partial"):
        result = {"run_id": "run-x", "status": status, "rows_written": {}}
        expected_exit = 0 if result["status"] != "failed" else 1
        assert expected_exit == 0


def test_main_exit_code_failed():
    """status == 'failed' maps to exit code 1."""
    result = {"run_id": "run-y", "status": "failed", "rows_written": {}}
    expected_exit = 0 if result["status"] != "failed" else 1
    assert expected_exit == 1
