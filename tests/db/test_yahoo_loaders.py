"""
tests/db/test_yahoo_loaders.py

Unit tests for src/db/loaders_yahoo.py.

All tests use in-memory DuckDB — no real MotherDuck connections are made.
"""

from __future__ import annotations

from collections.abc import Generator

import duckdb
import pandas as pd
import pytest

from src.db.loaders_yahoo import (
    load_matchups,
    load_players,
    load_rosters,
    load_transactions,
    stage_free_agents,
)
from src.db.schema import (
    DIM_PLAYERS,
    FACT_MATCHUPS,
    FACT_ROSTERS,
    FACT_TRANSACTIONS,
    FACT_WAIVER_SCORES,
    create_all_tables,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def conn() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Fresh in-memory DuckDB with all application tables created."""
    c = duckdb.connect(":memory:")
    create_all_tables(c)
    yield c
    c.close()


# ── load_rosters tests ────────────────────────────────────────────────────────


class TestLoadRosters:
    def _make_roster_df(self, rows: int = 2) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "team_id": [f"422.l.87941.t.{i}" for i in range(1, rows + 1)],
                "player_id": [f"422.p.100{i}" for i in range(rows)],
                "snapshot_date": ["2026-04-15"] * rows,
                "roster_slot": (["1B", "OF"] * rows)[:rows],
                "acquisition_type": (["draft", "waiver"] * rows)[:rows],
            }
        )

    def test_inserts_correct_row_count(self, conn: duckdb.DuckDBPyConnection) -> None:
        df = self._make_roster_df(rows=3)
        result = load_rosters(conn, df)
        assert result == 3

    def test_rows_persisted_in_db(self, conn: duckdb.DuckDBPyConnection) -> None:
        df = self._make_roster_df(rows=2)
        load_rosters(conn, df)
        count = conn.execute(f"SELECT COUNT(*) FROM {FACT_ROSTERS}").fetchone()
        assert count is not None
        assert count[0] == 2

    def test_idempotent_on_reinsertion(self, conn: duckdb.DuckDBPyConnection) -> None:
        df = self._make_roster_df(rows=2)
        load_rosters(conn, df)
        load_rosters(conn, df)  # same data again
        count = conn.execute(f"SELECT COUNT(*) FROM {FACT_ROSTERS}").fetchone()
        assert count is not None
        assert count[0] == 2  # still 2, not 4

    def test_upsert_updates_existing_row(self, conn: duckdb.DuckDBPyConnection) -> None:
        df1 = pd.DataFrame(
            {
                "team_id": ["422.l.87941.t.1"],
                "player_id": ["422.p.7578"],
                "snapshot_date": ["2026-04-15"],
                "roster_slot": ["BN"],
                "acquisition_type": ["draft"],
            }
        )
        df2 = pd.DataFrame(
            {
                "team_id": ["422.l.87941.t.1"],
                "player_id": ["422.p.7578"],
                "snapshot_date": ["2026-04-15"],
                "roster_slot": ["1B"],  # moved to active slot
                "acquisition_type": ["draft"],
            }
        )
        load_rosters(conn, df1)
        load_rosters(conn, df2)

        slot = conn.execute(
            f"SELECT roster_slot FROM {FACT_ROSTERS} WHERE player_id = '422.p.7578'"
        ).fetchone()
        assert slot is not None
        assert slot[0] == "1B"

    def test_empty_dataframe_returns_zero(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        empty = pd.DataFrame(
            columns=[
                "team_id",
                "player_id",
                "snapshot_date",
                "roster_slot",
                "acquisition_type",
            ]
        )
        result = load_rosters(conn, empty)
        assert result == 0

    def test_raises_when_required_column_missing(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        bad_df = pd.DataFrame(
            {
                "team_id": ["422.l.87941.t.1"],
                "player_id": ["422.p.7578"],
                # missing snapshot_date and roster_slot
            }
        )
        with pytest.raises(ValueError, match="snapshot_date"):
            load_rosters(conn, bad_df)

    def test_optional_acquisition_type_defaults_to_none(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        df = pd.DataFrame(
            {
                "team_id": ["422.l.87941.t.1"],
                "player_id": ["422.p.7578"],
                "snapshot_date": ["2026-04-15"],
                "roster_slot": ["SP"],
                # acquisition_type intentionally omitted
            }
        )
        load_rosters(conn, df)
        count = conn.execute(f"SELECT COUNT(*) FROM {FACT_ROSTERS}").fetchone()
        assert count is not None
        assert count[0] == 1


# ── load_transactions tests ───────────────────────────────────────────────────


class TestLoadTransactions:
    def _make_txn_df(self, rows: int = 2) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "transaction_id": [f"422.l.87941.tr.{i}" for i in range(rows)],
                "league_id": [87941] * rows,
                "transaction_date": ["2026-04-15T12:00:00+00:00"] * rows,
                "type": (["add", "drop"] * rows)[:rows],
                "team_id": [f"422.l.87941.t.{i}" for i in range(1, rows + 1)],
                "player_id": [f"422.p.200{i}" for i in range(rows)],
                "from_team_id": [None] * rows,
                "notes": [None] * rows,
            }
        )

    def test_inserts_correct_row_count(self, conn: duckdb.DuckDBPyConnection) -> None:
        df = self._make_txn_df(rows=3)
        result = load_transactions(conn, df)
        assert result == 3

    def test_rows_persisted_in_db(self, conn: duckdb.DuckDBPyConnection) -> None:
        df = self._make_txn_df(rows=2)
        load_transactions(conn, df)
        count = conn.execute(f"SELECT COUNT(*) FROM {FACT_TRANSACTIONS}").fetchone()
        assert count is not None
        assert count[0] == 2

    def test_idempotent_on_reinsertion(self, conn: duckdb.DuckDBPyConnection) -> None:
        df = self._make_txn_df(rows=2)
        load_transactions(conn, df)
        load_transactions(conn, df)
        count = conn.execute(f"SELECT COUNT(*) FROM {FACT_TRANSACTIONS}").fetchone()
        assert count is not None
        assert count[0] == 2

    def test_empty_dataframe_returns_zero(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        empty = pd.DataFrame(
            columns=[
                "transaction_id",
                "league_id",
                "transaction_date",
                "type",
                "team_id",
                "player_id",
            ]
        )
        result = load_transactions(conn, empty)
        assert result == 0

    def test_raises_when_required_column_missing(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        bad_df = pd.DataFrame(
            {
                "transaction_id": ["422.l.87941.tr.1"],
                # missing league_id, transaction_date, type, team_id, player_id
            }
        )
        with pytest.raises(ValueError, match="league_id"):
            load_transactions(conn, bad_df)

    def test_optional_columns_default_to_none(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        df = pd.DataFrame(
            {
                "transaction_id": ["422.l.87941.tr.99"],
                "league_id": [87941],
                "transaction_date": ["2026-04-15T12:00:00+00:00"],
                "type": ["add"],
                "team_id": ["422.l.87941.t.1"],
                "player_id": ["422.p.7578"],
                # from_team_id and notes intentionally omitted
            }
        )
        result = load_transactions(conn, df)
        assert result == 1


# ── load_players tests ────────────────────────────────────────────────────────


class TestLoadPlayers:
    def _make_players_df(self, rows: int = 2) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "player_id": [f"422.p.700{i}" for i in range(rows)],
                "mlb_id": [660000 + i for i in range(rows)],
                "fg_id": [f"fg{i}" for i in range(rows)],
                "full_name": [f"Player {i}" for i in range(rows)],
                "team": ["NYY", "BOS"][:rows],
                "positions": [["OF"], ["SS", "2B"]][:rows],
                "bats": ["L", "R"][:rows],
                "throws": ["R", "R"][:rows],
                "status": ["Active"] * rows,
                "updated_at": ["2026-04-15T00:00:00+00:00"] * rows,
            }
        )

    def test_inserts_correct_row_count(self, conn: duckdb.DuckDBPyConnection) -> None:
        df = self._make_players_df(rows=2)
        result = load_players(conn, df)
        assert result == 2

    def test_rows_persisted_in_db(self, conn: duckdb.DuckDBPyConnection) -> None:
        df = self._make_players_df(rows=2)
        load_players(conn, df)
        count = conn.execute(f"SELECT COUNT(*) FROM {DIM_PLAYERS}").fetchone()
        assert count is not None
        assert count[0] == 2

    def test_idempotent_on_reinsertion(self, conn: duckdb.DuckDBPyConnection) -> None:
        df = self._make_players_df(rows=2)
        load_players(conn, df)
        load_players(conn, df)
        count = conn.execute(f"SELECT COUNT(*) FROM {DIM_PLAYERS}").fetchone()
        assert count is not None
        assert count[0] == 2

    def test_upsert_updates_team(self, conn: duckdb.DuckDBPyConnection) -> None:
        df1 = pd.DataFrame(
            {
                "player_id": ["422.p.7578"],
                "full_name": ["Yordan Alvarez"],
                "team": ["HOU"],
            }
        )
        df2 = pd.DataFrame(
            {
                "player_id": ["422.p.7578"],
                "full_name": ["Yordan Alvarez"],
                "team": ["NYY"],  # traded!
            }
        )
        load_players(conn, df1)
        load_players(conn, df2)
        team = conn.execute(
            f"SELECT team FROM {DIM_PLAYERS} WHERE player_id = '422.p.7578'"
        ).fetchone()
        assert team is not None
        assert team[0] == "NYY"

    def test_empty_dataframe_returns_zero(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        empty = pd.DataFrame(columns=["player_id", "full_name"])
        result = load_players(conn, empty)
        assert result == 0

    def test_raises_when_full_name_missing(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        bad_df = pd.DataFrame({"player_id": ["422.p.7578"]})
        with pytest.raises(ValueError, match="full_name"):
            load_players(conn, bad_df)

    def test_raises_when_player_id_missing(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        bad_df = pd.DataFrame({"full_name": ["Yordan Alvarez"]})
        with pytest.raises(ValueError, match="player_id"):
            load_players(conn, bad_df)

    def test_optional_columns_get_defaults(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        df = pd.DataFrame(
            {
                "player_id": ["422.p.9001"],
                "full_name": ["Mystery Player"],
                # all other columns omitted
            }
        )
        result = load_players(conn, df)
        assert result == 1


# ── stage_free_agents tests ───────────────────────────────────────────────────


class TestStageFreeAgents:
    def _make_fa_df(self, rows: int = 3) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "player_id": [f"422.p.500{i}" for i in range(rows)],
                "full_name": [f"FA Player {i}" for i in range(rows)],
                "team": ["NYY"] * rows,
                "positions": [["OF"]] * rows,
                "status": ["Active"] * rows,
            }
        )

    def test_stages_correct_row_count(self, conn: duckdb.DuckDBPyConnection) -> None:
        df = self._make_fa_df(rows=3)
        result = stage_free_agents(conn, df)
        assert result == 3

    def test_rows_in_fact_waiver_scores(self, conn: duckdb.DuckDBPyConnection) -> None:
        df = self._make_fa_df(rows=2)
        stage_free_agents(conn, df)
        count = conn.execute(f"SELECT COUNT(*) FROM {FACT_WAIVER_SCORES}").fetchone()
        assert count is not None
        assert count[0] == 2

    def test_idempotent_on_restage(self, conn: duckdb.DuckDBPyConnection) -> None:
        df = self._make_fa_df(rows=2)
        stage_free_agents(conn, df)
        stage_free_agents(conn, df)
        count = conn.execute(f"SELECT COUNT(*) FROM {FACT_WAIVER_SCORES}").fetchone()
        assert count is not None
        assert count[0] == 2

    def test_empty_dataframe_returns_zero(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        empty = pd.DataFrame(columns=["player_id"])
        result = stage_free_agents(conn, empty)
        assert result == 0

    def test_raises_when_player_id_missing(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        bad_df = pd.DataFrame({"full_name": ["Someone"]})
        with pytest.raises(ValueError, match="player_id"):
            stage_free_agents(conn, bad_df)

    def test_overall_score_defaults_to_zero(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        df = pd.DataFrame({"player_id": ["422.p.5001"]})
        stage_free_agents(conn, df)
        row = conn.execute(
            f"SELECT overall_score FROM {FACT_WAIVER_SCORES} "
            "WHERE player_id = '422.p.5001'"
        ).fetchone()
        assert row is not None
        assert float(row[0]) == pytest.approx(0.0)


# ── load_matchups tests ──────────────────────────────────────────────────────


class TestLoadMatchups:
    def _make_matchup_df(self, rows: int = 1) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "matchup_id": [
                    f"87941_2026_W{i:02d}_T1vsT2" for i in range(1, rows + 1)
                ],
                "league_id": [87941] * rows,
                "week_number": list(range(1, rows + 1)),
                "season": [2026] * rows,
                "team_id_home": ["469.l.87941.t.1"] * rows,
                "team_id_away": ["469.l.87941.t.2"] * rows,
                "h_home": [42] * rows,
                "h_away": [38] * rows,
                "hr_home": [8] * rows,
                "hr_away": [9] * rows,
                "sb_home": [3] * rows,
                "sb_away": [1] * rows,
                "bb_home": [15] * rows,
                "bb_away": [12] * rows,
                "avg_home": [0.265] * rows,
                "avg_away": [0.248] * rows,
                "ops_home": [0.780] * rows,
                "ops_away": [0.720] * rows,
                "fpct_home": [0.985] * rows,
                "fpct_away": [0.980] * rows,
                "w_home": [3] * rows,
                "w_away": [2] * rows,
                "k_home": [55] * rows,
                "k_away": [48] * rows,
                "whip_home": [1.18] * rows,
                "whip_away": [1.32] * rows,
                "k_bb_home": [3.2] * rows,
                "k_bb_away": [2.8] * rows,
                "sv_h_home": [4] * rows,
                "sv_h_away": [3] * rows,
                "categories_won_home": [None] * rows,
                "categories_won_away": [None] * rows,
                "result": [None] * rows,
            }
        )

    def test_inserts_correct_row_count(self, conn: duckdb.DuckDBPyConnection) -> None:
        df = self._make_matchup_df(rows=2)
        result = load_matchups(conn, df)
        assert result == 2

    def test_rows_persisted_in_db(self, conn: duckdb.DuckDBPyConnection) -> None:
        df = self._make_matchup_df(rows=1)
        load_matchups(conn, df)
        count = conn.execute(f"SELECT COUNT(*) FROM {FACT_MATCHUPS}").fetchone()
        assert count is not None
        assert count[0] == 1

    def test_idempotent_on_reinsertion(self, conn: duckdb.DuckDBPyConnection) -> None:
        df = self._make_matchup_df(rows=1)
        load_matchups(conn, df)
        load_matchups(conn, df)
        count = conn.execute(f"SELECT COUNT(*) FROM {FACT_MATCHUPS}").fetchone()
        assert count is not None
        assert count[0] == 1

    def test_empty_dataframe_returns_zero(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        empty = pd.DataFrame(columns=["matchup_id"])
        result = load_matchups(conn, empty)
        assert result == 0

    def test_missing_matchup_id_returns_zero(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        df = pd.DataFrame({"league_id": [87941]})
        result = load_matchups(conn, df)
        assert result == 0

    def test_missing_optional_cols_filled_with_none(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        """DataFrame without outcome columns should still load successfully."""
        df = pd.DataFrame(
            {
                "matchup_id": ["87941_2026_W01_T1vsT2"],
                "league_id": [87941],
                "week_number": [1],
                "season": [2026],
                "team_id_home": ["469.l.87941.t.1"],
                "team_id_away": ["469.l.87941.t.2"],
                "h_home": [42],
                "h_away": [38],
                # Intentionally omit many stat and outcome columns
            }
        )
        result = load_matchups(conn, df)
        assert result == 1
        row = conn.execute(
            f"SELECT categories_won_home, result FROM {FACT_MATCHUPS}"
        ).fetchone()
        assert row is not None
        assert row[0] is None  # categories_won_home
        assert row[1] is None  # result
