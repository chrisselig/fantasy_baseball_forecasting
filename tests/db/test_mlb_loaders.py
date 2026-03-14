"""
tests/db/test_mlb_loaders.py

Unit tests for src/db/loaders_mlb.py.

Uses in-memory DuckDB — no real MotherDuck connections.
"""

from __future__ import annotations

import datetime

import duckdb
import pandas as pd
import pytest

from src.db.loaders_mlb import (
    get_fantasy_week,
    load_daily_stats,
    load_dim_dates,
    load_projections,
    load_weekly_stats,
    update_player_crosswalk,
)
from src.db.schema import (
    DIM_DATES,
    DIM_PLAYERS,
    FACT_PLAYER_STATS_DAILY,
    FACT_PLAYER_STATS_WEEKLY,
    FACT_PROJECTIONS,
    create_all_tables,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB connection with all tables created."""
    c = duckdb.connect(":memory:")
    create_all_tables(c)
    return c


def _make_daily_df(
    player_ids: list[str] | None = None,
    stat_date: datetime.date | None = None,
    **kwargs: object,
) -> pd.DataFrame:
    """Build a minimal daily stats DataFrame for testing."""
    if player_ids is None:
        player_ids = ["p1"]
    if stat_date is None:
        stat_date = datetime.date(2026, 4, 1)

    n = len(player_ids)
    data: dict[str, object] = {
        "player_id": player_ids,
        "stat_date": [stat_date] * n,
        "ab": kwargs.get("ab", [4] * n),
        "h": kwargs.get("h", [1] * n),
        "hr": kwargs.get("hr", [0] * n),
        "sb": kwargs.get("sb", [0] * n),
        "bb": kwargs.get("bb", [1] * n),
        "hbp": kwargs.get("hbp", [0] * n),
        "sf": kwargs.get("sf", [0] * n),
        "tb": kwargs.get("tb", [1] * n),
        "errors": kwargs.get("errors", [0] * n),
        "chances": kwargs.get("chances", [3] * n),
        "ip": kwargs.get("ip", [0.0] * n),
        "w": kwargs.get("w", [0] * n),
        "k": kwargs.get("k", [0] * n),
        "walks_allowed": kwargs.get("walks_allowed", [0] * n),
        "hits_allowed": kwargs.get("hits_allowed", [0] * n),
        "sv": kwargs.get("sv", [0] * n),
        "holds": kwargs.get("holds", [0] * n),
    }
    return pd.DataFrame(data)


def _make_weekly_input_df(
    player_ids: list[str],
    week_number: int,
    season: int,
    **kwargs: object,
) -> pd.DataFrame:
    """Build a minimal weekly-input DataFrame (already grouped)."""
    n = len(player_ids)
    data: dict[str, object] = {
        "player_id": player_ids,
        "week_number": [week_number] * n,
        "season": [season] * n,
        "ab": kwargs.get("ab", [20] * n),
        "h": kwargs.get("h", [6] * n),
        "hr": kwargs.get("hr", [1] * n),
        "sb": kwargs.get("sb", [1] * n),
        "bb": kwargs.get("bb", [3] * n),
        "hbp": kwargs.get("hbp", [0] * n),
        "sf": kwargs.get("sf", [0] * n),
        "tb": kwargs.get("tb", [9] * n),
        "errors": kwargs.get("errors", [0] * n),
        "chances": kwargs.get("chances", [15] * n),
        "ip": kwargs.get("ip", [0.0] * n),
        "w": kwargs.get("w", [0] * n),
        "k": kwargs.get("k", [0] * n),
        "walks_allowed": kwargs.get("walks_allowed", [0] * n),
        "hits_allowed": kwargs.get("hits_allowed", [0] * n),
        "sv": kwargs.get("sv", [0] * n),
        "holds": kwargs.get("holds", [0] * n),
    }
    return pd.DataFrame(data)


# ── load_daily_stats tests ────────────────────────────────────────────────────


class TestLoadDailyStats:
    def test_inserts_correct_rows(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Inserts the correct number of rows."""
        df = _make_daily_df(player_ids=["p1", "p2"])
        count = load_daily_stats(conn, df, datetime.date(2026, 4, 1))

        assert count == 2
        result = conn.execute(
            f"SELECT COUNT(*) FROM {FACT_PLAYER_STATS_DAILY}"
        ).fetchone()
        assert result is not None
        assert result[0] == 2

    def test_is_idempotent(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Second insert of same rows replaces — does not duplicate."""
        df = _make_daily_df(player_ids=["p1"])
        load_daily_stats(conn, df, datetime.date(2026, 4, 1))
        load_daily_stats(conn, df, datetime.date(2026, 4, 1))

        result = conn.execute(
            f"SELECT COUNT(*) FROM {FACT_PLAYER_STATS_DAILY}"
        ).fetchone()
        assert result is not None
        assert result[0] == 1

    def test_stat_date_overridden(self, conn: duckdb.DuckDBPyConnection) -> None:
        """stat_date column in the table matches the provided stat_date argument."""
        df = _make_daily_df(player_ids=["p1"])
        target_date = datetime.date(2026, 4, 5)
        load_daily_stats(conn, df, target_date)

        row = conn.execute(
            f"SELECT stat_date FROM {FACT_PLAYER_STATS_DAILY} WHERE player_id='p1'"
        ).fetchone()
        assert row is not None
        assert row[0] == target_date

    def test_raises_on_missing_required_columns(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        """Raises ValueError when required columns are absent."""
        bad_df = pd.DataFrame({"ab": [4], "h": [1]})  # missing player_id, stat_date

        with pytest.raises(ValueError, match="player_id"):
            load_daily_stats(conn, bad_df, datetime.date(2026, 4, 1))

    def test_empty_df_returns_zero(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Empty DataFrame returns 0 without inserting."""
        df = pd.DataFrame(columns=["player_id", "stat_date"])
        count = load_daily_stats(conn, df, datetime.date(2026, 4, 1))

        assert count == 0
        result = conn.execute(
            f"SELECT COUNT(*) FROM {FACT_PLAYER_STATS_DAILY}"
        ).fetchone()
        assert result is not None
        assert result[0] == 0

    def test_multiple_dates_accumulate(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Stats for different dates accumulate (not replaced)."""
        df1 = _make_daily_df(player_ids=["p1"])
        df2 = _make_daily_df(player_ids=["p1"])
        load_daily_stats(conn, df1, datetime.date(2026, 4, 1))
        load_daily_stats(conn, df2, datetime.date(2026, 4, 2))

        result = conn.execute(
            f"SELECT COUNT(*) FROM {FACT_PLAYER_STATS_DAILY}"
        ).fetchone()
        assert result is not None
        assert result[0] == 2


# ── load_weekly_stats tests ───────────────────────────────────────────────────


class TestLoadWeeklyStats:
    def test_inserts_rows(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Inserts the correct number of rows."""
        df = _make_weekly_input_df(["p1", "p2"], week_number=1, season=2026)
        count = load_weekly_stats(conn, df, week_number=1, season=2026)

        assert count == 2
        result = conn.execute(
            f"SELECT COUNT(*) FROM {FACT_PLAYER_STATS_WEEKLY}"
        ).fetchone()
        assert result is not None
        assert result[0] == 2

    def test_rate_stats_computed_from_components(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        """avg, ops, whip, fpct computed from SUM(components) not AVG(rates)."""
        # Two daily rows for the same player — loader aggregates them
        df = pd.DataFrame(
            {
                "player_id": ["p1", "p1"],
                "week_number": [1, 1],
                "season": [2026, 2026],
                "ab": [4, 4],  # SUM(ab) = 8
                "h": [2, 1],  # SUM(h) = 3  → avg = 3/8 = 0.375
                "hr": [1, 0],
                "sb": [0, 1],
                "bb": [1, 0],  # SUM(bb) = 1
                "hbp": [0, 0],
                "sf": [0, 0],
                "tb": [5, 2],  # SUM(tb) = 7  → SLG = 7/8
                "errors": [0, 0],
                "chances": [3, 3],  # SUM(chances) = 6
                "ip": [0.0, 0.0],
                "w": [0, 0],
                "k": [0, 0],
                "walks_allowed": [0, 0],
                "hits_allowed": [0, 0],
                "sv": [0, 0],
                "holds": [0, 0],
            }
        )

        load_weekly_stats(conn, df, week_number=1, season=2026)

        row = conn.execute(
            f"SELECT avg, ops FROM {FACT_PLAYER_STATS_WEEKLY} WHERE player_id='p1'"
        ).fetchone()
        assert row is not None
        avg, ops = row

        # avg = 3/8 = 0.375
        assert pytest.approx(float(avg), rel=1e-3) == 0.375

        # OBP = (3 + 1 + 0) / (8 + 1 + 0 + 0) = 4/9
        # SLG = 7/8
        expected_ops = 7 / 8 + 4 / 9
        assert pytest.approx(float(ops), rel=1e-3) == expected_ops

    def test_whip_computed_from_components(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        """WHIP = (SUM(walks_allowed) + SUM(hits_allowed)) / SUM(ip) [LOWEST WINS]."""
        df = pd.DataFrame(
            {
                "player_id": ["p99", "p99"],
                "week_number": [1, 1],
                "season": [2026, 2026],
                "ab": [0, 0],
                "h": [0, 0],
                "hr": [0, 0],
                "sb": [0, 0],
                "bb": [0, 0],
                "hbp": [0, 0],
                "sf": [0, 0],
                "tb": [0, 0],
                "errors": [0, 0],
                "chances": [0, 0],
                "ip": [3.0, 3.0],  # SUM(ip) = 6
                "w": [1, 0],
                "k": [5, 4],
                "walks_allowed": [1, 2],  # SUM(wa) = 3
                "hits_allowed": [3, 3],  # SUM(ha) = 6
                "sv": [0, 0],
                "holds": [0, 0],
            }
        )

        load_weekly_stats(conn, df, week_number=1, season=2026)

        row = conn.execute(
            f"SELECT whip FROM {FACT_PLAYER_STATS_WEEKLY} WHERE player_id='p99'"
        ).fetchone()
        assert row is not None

        # WHIP = (3 + 6) / 6 = 1.5  [LOWEST WINS]
        assert pytest.approx(float(row[0]), rel=1e-3) == 1.5

    def test_k_bb_handles_zero_walks_allowed(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        """k_bb is None (not a division error) when walks_allowed = 0."""
        df = _make_weekly_input_df(
            ["pitcher1"],
            week_number=2,
            season=2026,
            k=[10],
            walks_allowed=[0],  # Division by zero scenario
            ip=[7.0],
        )

        load_weekly_stats(conn, df, week_number=2, season=2026)

        row = conn.execute(
            f"SELECT k_bb FROM {FACT_PLAYER_STATS_WEEKLY} WHERE player_id='pitcher1'"
        ).fetchone()
        assert row is not None
        assert row[0] is None  # Not a division error

    def test_sv_h_is_sv_plus_holds(self, conn: duckdb.DuckDBPyConnection) -> None:
        """sv_h = SUM(sv) + SUM(holds)."""
        df = _make_weekly_input_df(
            ["closer1"],
            week_number=3,
            season=2026,
            sv=[2],
            holds=[3],
        )

        load_weekly_stats(conn, df, week_number=3, season=2026)

        row = conn.execute(
            f"SELECT sv_h FROM {FACT_PLAYER_STATS_WEEKLY} WHERE player_id='closer1'"
        ).fetchone()
        assert row is not None
        assert row[0] == 5

    def test_fpct_computed_from_components(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        """fpct = (SUM(chances) - SUM(errors)) / SUM(chances)."""
        df = _make_weekly_input_df(
            ["fielder1"],
            week_number=1,
            season=2026,
            errors=[2],
            chances=[20],
        )

        load_weekly_stats(conn, df, week_number=1, season=2026)

        row = conn.execute(
            f"SELECT fpct FROM {FACT_PLAYER_STATS_WEEKLY} WHERE player_id='fielder1'"
        ).fetchone()
        assert row is not None
        # fpct = (20 - 2) / 20 = 0.9
        assert pytest.approx(float(row[0]), rel=1e-3) == 0.9

    def test_raises_on_missing_required_columns(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        """Raises ValueError when required columns are missing."""
        bad_df = pd.DataFrame(
            {"player_id": ["p1"], "ab": [10]}
        )  # missing week_number, season

        with pytest.raises(ValueError, match="week_number"):
            load_weekly_stats(conn, bad_df, week_number=1, season=2026)

    def test_empty_df_returns_zero(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Empty DataFrame returns 0."""
        df = pd.DataFrame(columns=["player_id", "week_number", "season"])
        count = load_weekly_stats(conn, df, week_number=1, season=2026)
        assert count == 0

    def test_rebuild_replaces_existing_rows(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        """Second call for same week/season replaces the previous rows."""
        df1 = _make_weekly_input_df(["p1"], week_number=1, season=2026, h=[3])
        df2 = _make_weekly_input_df(["p1"], week_number=1, season=2026, h=[6])

        load_weekly_stats(conn, df1, week_number=1, season=2026)
        load_weekly_stats(conn, df2, week_number=1, season=2026)

        result = conn.execute(
            f"SELECT COUNT(*) FROM {FACT_PLAYER_STATS_WEEKLY}"
        ).fetchone()
        assert result is not None
        assert result[0] == 1  # Not doubled


# ── load_projections tests ────────────────────────────────────────────────────


class TestLoadProjections:
    def _make_proj_df(self, player_ids: list[str] | None = None) -> pd.DataFrame:
        if player_ids is None:
            player_ids = ["p1"]
        n = len(player_ids)
        return pd.DataFrame(
            {
                "player_id": player_ids,
                "projection_date": [datetime.date(2026, 4, 1)] * n,
                "target_week": [1] * n,
                "source": ["steamer"] * n,
                "proj_h": [150.0] * n,
                "proj_hr": [20.0] * n,
            }
        )

    def test_inserts_rows(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Inserts and returns correct row count."""
        df = self._make_proj_df(["p1", "p2"])
        count = load_projections(conn, df)

        assert count == 2
        result = conn.execute(f"SELECT COUNT(*) FROM {FACT_PROJECTIONS}").fetchone()
        assert result is not None
        assert result[0] == 2

    def test_is_idempotent(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Double insert replaces — does not duplicate."""
        df = self._make_proj_df(["p1"])
        load_projections(conn, df)
        load_projections(conn, df)

        result = conn.execute(f"SELECT COUNT(*) FROM {FACT_PROJECTIONS}").fetchone()
        assert result is not None
        assert result[0] == 1

    def test_raises_on_missing_required_columns(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        """Raises ValueError when required columns are missing."""
        bad_df = pd.DataFrame({"player_id": ["p1"], "proj_h": [100.0]})

        with pytest.raises(ValueError, match="projection_date"):
            load_projections(conn, bad_df)

    def test_empty_df_returns_zero(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Empty DataFrame returns 0."""
        df = pd.DataFrame(
            columns=["player_id", "projection_date", "target_week", "source"]
        )
        count = load_projections(conn, df)
        assert count == 0


# ── load_dim_dates tests ──────────────────────────────────────────────────────


class TestLoadDimDates:
    def test_inserts_correct_number_of_rows(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        """Inserts one row per calendar date in range."""
        start = datetime.date(2026, 4, 1)
        end = datetime.date(2026, 4, 7)  # 7 days
        count = load_dim_dates(conn, season=2026, start_date=start, end_date=end)

        assert count == 7
        result = conn.execute(f"SELECT COUNT(*) FROM {DIM_DATES}").fetchone()
        assert result is not None
        assert result[0] == 7

    def test_week_number_computed_correctly(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        """week_number is derived relative to season start."""
        # Season start 2026 = March 26 (Thursday). Week 1 anchor = Monday March 23.
        # April 1 = 9 days after March 23 → week 2
        start = datetime.date(2026, 4, 1)
        end = datetime.date(2026, 4, 1)
        load_dim_dates(conn, season=2026, start_date=start, end_date=end)

        row = conn.execute(
            f"SELECT week_number FROM {DIM_DATES} WHERE date = '2026-04-01'"
        ).fetchone()
        assert row is not None
        # Week 1 anchor: March 23 (Monday before March 26)
        # April 1 is day index 9 from March 23 → week 2
        assert row[0] == 2

    def test_day_of_week_correct(self, conn: duckdb.DuckDBPyConnection) -> None:
        """day_of_week is the full weekday name."""
        start = datetime.date(2026, 4, 6)  # Monday
        end = datetime.date(2026, 4, 6)
        load_dim_dates(conn, season=2026, start_date=start, end_date=end)

        row = conn.execute(
            f"SELECT day_of_week FROM {DIM_DATES} WHERE date = '2026-04-06'"
        ).fetchone()
        assert row is not None
        assert row[0] == "Monday"

    def test_is_playoff_week_defaults_false(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        """is_playoff_week defaults to False for all inserted rows."""
        load_dim_dates(
            conn,
            season=2026,
            start_date=datetime.date(2026, 4, 1),
            end_date=datetime.date(2026, 4, 3),
        )

        rows = conn.execute(f"SELECT is_playoff_week FROM {DIM_DATES}").fetchall()
        assert all(not row[0] for row in rows)

    def test_inverted_range_returns_zero(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Returns 0 when start_date > end_date."""
        count = load_dim_dates(
            conn,
            season=2026,
            start_date=datetime.date(2026, 4, 10),
            end_date=datetime.date(2026, 4, 1),
        )
        assert count == 0


# ── get_fantasy_week tests ────────────────────────────────────────────────────


class TestGetFantasyWeek:
    def test_season_start_day_is_week_1(self) -> None:
        """The season start date itself is in week 1."""
        season_start = datetime.date(2026, 3, 26)  # Thursday
        # Week 1 anchor = Monday March 23
        assert get_fantasy_week(season_start, season_start) == 1

    def test_day_before_season_is_week_1(self) -> None:
        """Dates before season start clamp to week 1."""
        season_start = datetime.date(2026, 3, 26)
        before = datetime.date(2026, 3, 20)
        assert get_fantasy_week(before, season_start) == 1

    def test_second_monday_is_week_2(self) -> None:
        """The Monday 7 days after the week 1 anchor is week 2."""
        season_start = datetime.date(2026, 3, 26)
        # Week 1 anchor = March 23 (Monday)
        week2_start = datetime.date(2026, 3, 30)  # Monday, 7 days after March 23
        assert get_fantasy_week(week2_start, season_start) == 2

    def test_week_progression(self) -> None:
        """Week numbers increment weekly."""
        season_start = datetime.date(2026, 3, 26)
        # Week 1 anchor: March 23
        dates_and_weeks = [
            (datetime.date(2026, 3, 23), 1),
            (datetime.date(2026, 3, 29), 1),
            (datetime.date(2026, 3, 30), 2),
            (datetime.date(2026, 4, 5), 2),
            (datetime.date(2026, 4, 6), 3),
        ]
        for date, expected_week in dates_and_weeks:
            assert get_fantasy_week(date, season_start) == expected_week, (
                f"Expected week {expected_week} for {date}"
            )


# ── update_player_crosswalk tests ─────────────────────────────────────────────


class TestUpdatePlayerCrosswalk:
    def _insert_player(
        self,
        conn: duckdb.DuckDBPyConnection,
        player_id: str,
        full_name: str,
    ) -> None:
        """Insert a minimal dim_players row."""
        conn.execute(
            f"""
            INSERT INTO {DIM_PLAYERS} (player_id, full_name, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
            [player_id, full_name],
        )

    def test_updates_mlb_id_and_fg_id(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Updates mlb_id and fg_id on matching dim_players rows."""
        self._insert_player(conn, "yahoo.p.001", "John Doe")

        crosswalk = pd.DataFrame(
            {"full_name": ["John Doe"], "mlb_id": [12345], "fg_id": ["fg001"]}
        )
        update_player_crosswalk(conn, crosswalk)

        row = conn.execute(
            f"SELECT mlb_id, fg_id FROM {DIM_PLAYERS} WHERE player_id='yahoo.p.001'"
        ).fetchone()
        assert row is not None
        assert row[0] == 12345
        assert row[1] == "fg001"

    def test_raises_on_missing_full_name_column(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        """Raises ValueError if full_name column is missing."""
        bad_df = pd.DataFrame({"mlb_id": [12345], "fg_id": ["fg001"]})

        with pytest.raises(ValueError, match="full_name"):
            update_player_crosswalk(conn, bad_df)

    def test_empty_df_returns_zero(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Empty DataFrame returns 0 without error."""
        df = pd.DataFrame(columns=["full_name", "mlb_id", "fg_id"])
        count = update_player_crosswalk(conn, df)
        assert count == 0

    def test_no_match_does_not_error(self, conn: duckdb.DuckDBPyConnection) -> None:
        """No-op when full_name has no match in dim_players."""
        crosswalk = pd.DataFrame(
            {"full_name": ["Unknown Player"], "mlb_id": [99999], "fg_id": ["fgXXX"]}
        )
        count = update_player_crosswalk(conn, crosswalk)
        assert count >= 0  # No error, no crash

    def test_rows_without_mlb_id_not_processed(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        """Rows where mlb_id is None are not used to update dim_players."""
        self._insert_player(conn, "yahoo.p.002", "Jane Smith")

        crosswalk = pd.DataFrame(
            {"full_name": ["Jane Smith"], "mlb_id": [None], "fg_id": ["fgXXX"]}
        )
        update_player_crosswalk(conn, crosswalk)

        row = conn.execute(
            f"SELECT mlb_id FROM {DIM_PLAYERS} WHERE player_id='yahoo.p.002'"
        ).fetchone()
        assert row is not None
        assert row[0] is None  # Should remain NULL
