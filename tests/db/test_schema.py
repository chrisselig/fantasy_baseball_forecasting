"""Tests for src/db/schema.py."""

from __future__ import annotations

from collections.abc import Generator

import duckdb
import pytest

from src.db.schema import (
    ALL_TABLES,
    DIM_DATES,
    DIM_PLAYERS,
    FACT_DAILY_REPORTS,
    FACT_MATCHUPS,
    FACT_PIPELINE_RUNS,
    FACT_PLAYER_STATS_DAILY,
    FACT_PLAYER_STATS_WEEKLY,
    FACT_PROJECTIONS,
    FACT_ROSTERS,
    FACT_TRANSACTIONS,
    FACT_WAIVER_SCORES,
    create_all_tables,
    drop_all_tables,
    get_existing_tables,
)


@pytest.fixture()
def mem_conn() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Provide a fresh in-memory DuckDB connection for each test."""
    conn = duckdb.connect(":memory:")
    yield conn
    conn.close()


class TestTableConstants:
    def test_all_tables_has_eleven_entries(self) -> None:
        assert len(ALL_TABLES) == 11

    def test_all_expected_tables_present(self) -> None:
        expected = {
            DIM_PLAYERS,
            DIM_DATES,
            FACT_PLAYER_STATS_DAILY,
            FACT_PLAYER_STATS_WEEKLY,
            FACT_ROSTERS,
            FACT_TRANSACTIONS,
            FACT_MATCHUPS,
            FACT_WAIVER_SCORES,
            FACT_PROJECTIONS,
            FACT_DAILY_REPORTS,
            FACT_PIPELINE_RUNS,
        }
        assert set(ALL_TABLES) == expected


class TestCreateAllTables:
    def test_creates_all_tables(self, mem_conn: duckdb.DuckDBPyConnection) -> None:
        create_all_tables(mem_conn)
        existing = get_existing_tables(mem_conn)
        for table in ALL_TABLES:
            assert table in existing, f"Table '{table}' was not created"

    def test_idempotent_on_second_call(
        self, mem_conn: duckdb.DuckDBPyConnection
    ) -> None:
        create_all_tables(mem_conn)
        # Should not raise — IF NOT EXISTS prevents errors
        create_all_tables(mem_conn)
        assert len(get_existing_tables(mem_conn)) == len(ALL_TABLES)


class TestDropAllTables:
    def test_drops_all_tables(self, mem_conn: duckdb.DuckDBPyConnection) -> None:
        create_all_tables(mem_conn)
        drop_all_tables(mem_conn)
        assert get_existing_tables(mem_conn) == []

    def test_idempotent_on_empty_db(self, mem_conn: duckdb.DuckDBPyConnection) -> None:
        # Should not raise on an empty database
        drop_all_tables(mem_conn)
        assert get_existing_tables(mem_conn) == []


class TestTableStructure:
    """Spot-check that key columns exist and have the right types."""

    def test_dim_players_has_fg_id(self, mem_conn: duckdb.DuckDBPyConnection) -> None:
        create_all_tables(mem_conn)
        mem_conn.execute(
            f"INSERT INTO {DIM_PLAYERS} "
            "(player_id, fg_id, full_name) VALUES ('p1', 'fg123', 'Test Player')"
        )
        row = mem_conn.execute(
            f"SELECT fg_id FROM {DIM_PLAYERS} WHERE player_id = 'p1'"
        ).fetchone()
        assert row is not None
        assert row[0] == "fg123"

    def test_fact_player_stats_daily_has_ops_components(
        self, mem_conn: duckdb.DuckDBPyConnection
    ) -> None:
        create_all_tables(mem_conn)
        mem_conn.execute(
            f"INSERT INTO {FACT_PLAYER_STATS_DAILY} "
            "(player_id, stat_date, ab, h, hbp, sf, tb) "
            "VALUES ('p1', '2026-04-01', 4, 2, 1, 0, 3)"
        )
        row = mem_conn.execute(
            f"SELECT ab, h, hbp, sf, tb FROM {FACT_PLAYER_STATS_DAILY}"
        ).fetchone()
        assert row == (4, 2, 1, 0, 3)

    def test_fact_player_stats_weekly_has_components(
        self, mem_conn: duckdb.DuckDBPyConnection
    ) -> None:
        create_all_tables(mem_conn)
        mem_conn.execute(
            f"INSERT INTO {FACT_PLAYER_STATS_WEEKLY} "
            "(player_id, week_number, season, ab, hbp, sf, tb, errors, chances) "
            "VALUES ('p1', 1, 2026, 20, 2, 1, 8, 0, 15)"
        )
        row = mem_conn.execute(
            f"SELECT ab, hbp, sf, tb FROM {FACT_PLAYER_STATS_WEEKLY}"
        ).fetchone()
        assert row == (20, 2, 1, 8)

    def test_fact_projections_has_component_columns(
        self, mem_conn: duckdb.DuckDBPyConnection
    ) -> None:
        create_all_tables(mem_conn)
        mem_conn.execute(
            f"INSERT INTO {FACT_PROJECTIONS} "
            "(player_id, projection_date, target_week, proj_ab, proj_tb, "
            "proj_walks_allowed, proj_hits_allowed, source) "
            "VALUES ('p1', '2026-04-01', 1, 15.0, 6.0, 2.0, 8.0, 'steamer')"
        )
        row = mem_conn.execute(
            f"SELECT proj_ab, proj_tb, proj_walks_allowed, proj_hits_allowed "
            f"FROM {FACT_PROJECTIONS}"
        ).fetchone()
        assert row == (15.0, 6.0, 2.0, 8.0)

    def test_fact_daily_reports_exists(
        self, mem_conn: duckdb.DuckDBPyConnection
    ) -> None:
        create_all_tables(mem_conn)
        assert FACT_DAILY_REPORTS in get_existing_tables(mem_conn)

    def test_fact_pipeline_runs_exists(
        self, mem_conn: duckdb.DuckDBPyConnection
    ) -> None:
        create_all_tables(mem_conn)
        assert FACT_PIPELINE_RUNS in get_existing_tables(mem_conn)

    def test_whip_comment_documented_in_matchups(
        self, mem_conn: duckdb.DuckDBPyConnection
    ) -> None:
        """WHIP columns exist in matchups table (lowest wins)."""
        create_all_tables(mem_conn)
        mem_conn.execute(
            f"INSERT INTO {FACT_MATCHUPS} "
            "(matchup_id, league_id, week_number, season, "
            "team_id_home, team_id_away, whip_home, whip_away) "
            "VALUES ('m1', 87941, 1, 2026, 't1', 't2', 1.15, 1.32)"
        )
        row = mem_conn.execute(
            f"SELECT whip_home, whip_away FROM {FACT_MATCHUPS}"
        ).fetchone()
        assert row is not None
        assert float(row[0]) == pytest.approx(1.15)
        assert float(row[1]) == pytest.approx(1.32)
