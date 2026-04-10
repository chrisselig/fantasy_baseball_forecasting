"""
tests/db/test_loaders_advanced.py

Unit tests for src/db/loaders_advanced.py.
"""

from __future__ import annotations

import datetime

import duckdb
import pandas as pd
import pytest

from src.db.loaders_advanced import (
    _compute_batter_derived,
    _compute_pitcher_derived,
    load_advanced_stats,
)
from src.db.schema import (
    DIM_PLAYERS,
    FACT_PLAYER_ADVANCED_STATS,
    FACT_PLAYER_STATS_DAILY,
    create_all_tables,
)


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    create_all_tables(c)
    return c


def _seed_dim_players(
    conn: duckdb.DuckDBPyConnection,
    rows: list[tuple[str, int, list[str]]],
) -> None:
    now = datetime.datetime.now(datetime.UTC)
    df = pd.DataFrame(
        [
            {
                "player_id": pid,
                "mlb_id": mlb,
                "fg_id": None,
                "full_name": f"Player {pid}",
                "team": "NYY",
                "positions": pos,
                "bats": "R",
                "throws": "R",
                "status": "Active",
                "updated_at": now,
            }
            for pid, mlb, pos in rows
        ]
    )
    conn.register("_tmp_dp", df)
    conn.execute(f"INSERT INTO {DIM_PLAYERS} SELECT * FROM _tmp_dp")
    conn.unregister("_tmp_dp")


def _seed_daily(conn: duckdb.DuckDBPyConnection, rows: list[dict]) -> None:
    cols = [
        c[0] for c in conn.execute(f"DESCRIBE {FACT_PLAYER_STATS_DAILY}").fetchall()
    ]
    normalized = []
    for r in rows:
        row = dict.fromkeys(cols)
        row.update(r)
        normalized.append(row)
    df = pd.DataFrame(normalized)
    conn.register("_tmp_ds", df)
    conn.execute(f"INSERT INTO {FACT_PLAYER_STATS_DAILY} SELECT * FROM _tmp_ds")
    conn.unregister("_tmp_ds")


class TestComputeBatterDerived:
    def test_woba_computation_handles_ab_only(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        """wOBA should compute without error for players with only AB > 0."""
        _seed_daily(
            conn,
            [
                {
                    "player_id": "p1",
                    "stat_date": datetime.date(2024, 7, 1),
                    "ab": 4,
                    "h": 2,
                    "hr": 0,
                    "bb": 0,
                    "hbp": 0,
                    "sf": 0,
                    "tb": 3,
                }
            ],
        )
        out = _compute_batter_derived(conn, 2024)
        assert not out.empty
        assert "woba" in out.columns
        assert out.iloc[0]["player_id"] == "p1"
        assert out.iloc[0]["woba"] > 0

    def test_skips_players_with_zero_ab(self, conn: duckdb.DuckDBPyConnection) -> None:
        _seed_daily(
            conn,
            [
                {
                    "player_id": "p1",
                    "stat_date": datetime.date(2024, 7, 1),
                    "ab": 0,
                }
            ],
        )
        out = _compute_batter_derived(conn, 2024)
        assert out.empty


class TestComputePitcherDerived:
    def test_k_bb_pct_computed(self, conn: duckdb.DuckDBPyConnection) -> None:
        _seed_daily(
            conn,
            [
                {
                    "player_id": "p2",
                    "stat_date": datetime.date(2024, 7, 1),
                    "ip": 9.0,
                    "k": 12,
                    "walks_allowed": 2,
                    "hits_allowed": 5,
                }
            ],
        )
        out = _compute_pitcher_derived(conn, 2024)
        assert not out.empty
        row = out.iloc[0]
        assert row["player_id"] == "p2"
        # bf = 27 + 5 + 2 = 34; k_bb_pct = (12 - 2) / 34 * 100 ≈ 29.41
        assert row["k_bb_pct"] == pytest.approx(29.41, abs=0.05)

    def test_skips_players_with_zero_ip(self, conn: duckdb.DuckDBPyConnection) -> None:
        _seed_daily(
            conn,
            [
                {
                    "player_id": "p1",
                    "stat_date": datetime.date(2024, 7, 1),
                    "ip": 0.0,
                }
            ],
        )
        out = _compute_pitcher_derived(conn, 2024)
        assert out.empty


class TestLoadAdvancedStats:
    def test_end_to_end_merge_and_upsert(self, conn: duckdb.DuckDBPyConnection) -> None:
        _seed_dim_players(
            conn,
            [
                ("422.p.1", 444482, ["2B"]),
                ("422.p.2", 657277, ["SP"]),
            ],
        )
        _seed_daily(
            conn,
            [
                {
                    "player_id": "422.p.1",
                    "stat_date": datetime.date(2024, 7, 1),
                    "ab": 4,
                    "h": 2,
                    "hr": 1,
                    "bb": 1,
                    "hbp": 0,
                    "sf": 0,
                    "tb": 5,
                },
                {
                    "player_id": "422.p.2",
                    "stat_date": datetime.date(2024, 7, 1),
                    "ip": 6.0,
                    "k": 8,
                    "walks_allowed": 2,
                    "hits_allowed": 5,
                },
            ],
        )
        batter_savant = pd.DataFrame(
            [
                {
                    "mlb_id": 444482,
                    "season": 2024,
                    "xwoba": 0.380,
                    "barrel_pct": 10.5,
                    "hard_hit_pct": 45.2,
                    "avg_launch_angle": 12.3,
                    "sweet_spot_pct": 35.0,
                    "bat_speed_pctile": 75.0,
                    "sprint_speed_pctile": 50.0,
                }
            ]
        )
        pitcher_savant = pd.DataFrame(
            [
                {
                    "mlb_id": 657277,
                    "season": 2024,
                    "xera": 3.50,
                    "xwoba_against": 0.290,
                    "barrel_pct_against": 6.5,
                    "hard_hit_pct_against": 32.0,
                }
            ]
        )

        n = load_advanced_stats(conn, 2024, batter_savant, pitcher_savant)
        assert n == 2

        rows = conn.execute(
            f"SELECT player_id, xwoba, woba, xera, k_bb_pct "
            f"FROM {FACT_PLAYER_ADVANCED_STATS} ORDER BY player_id"
        ).fetchall()
        by_id = {r[0]: r for r in rows}
        assert "422.p.1" in by_id
        assert "422.p.2" in by_id
        # Batter row: xwoba=0.380 from savant, woba computed > 0
        assert float(by_id["422.p.1"][1]) == pytest.approx(0.380, abs=0.001)
        assert by_id["422.p.1"][2] is not None
        # Pitcher row: xera=3.50 from savant, k_bb_pct computed
        assert float(by_id["422.p.2"][3]) == pytest.approx(3.50, abs=0.01)
        assert by_id["422.p.2"][4] is not None

    def test_is_idempotent(self, conn: duckdb.DuckDBPyConnection) -> None:
        _seed_dim_players(conn, [("422.p.1", 444482, ["2B"])])
        _seed_daily(
            conn,
            [
                {
                    "player_id": "422.p.1",
                    "stat_date": datetime.date(2024, 7, 1),
                    "ab": 4,
                    "h": 2,
                    "hr": 1,
                    "bb": 1,
                    "hbp": 0,
                    "sf": 0,
                    "tb": 5,
                }
            ],
        )
        batter = pd.DataFrame(
            [
                {
                    "mlb_id": 444482,
                    "season": 2024,
                    "xwoba": 0.400,
                    "barrel_pct": None,
                    "hard_hit_pct": None,
                    "avg_launch_angle": None,
                    "sweet_spot_pct": None,
                    "bat_speed_pctile": None,
                    "sprint_speed_pctile": None,
                }
            ]
        )
        load_advanced_stats(conn, 2024, batter, pd.DataFrame())
        load_advanced_stats(conn, 2024, batter, pd.DataFrame())
        count = conn.execute(
            f"SELECT COUNT(*) FROM {FACT_PLAYER_ADVANCED_STATS}"
        ).fetchone()[0]
        assert count == 1

    def test_drops_unknown_mlb_ids(self, conn: duckdb.DuckDBPyConnection) -> None:
        _seed_dim_players(conn, [("422.p.1", 444482, ["2B"])])
        # mlb_id 999999 doesn't exist in dim_players
        batter = pd.DataFrame(
            [
                {
                    "mlb_id": 999999,
                    "season": 2024,
                    "xwoba": 0.400,
                    "barrel_pct": None,
                    "hard_hit_pct": None,
                    "avg_launch_angle": None,
                    "sweet_spot_pct": None,
                    "bat_speed_pctile": None,
                    "sprint_speed_pctile": None,
                }
            ]
        )
        n = load_advanced_stats(conn, 2024, batter, pd.DataFrame())
        assert n == 0

    def test_handles_all_empty_inputs(self, conn: duckdb.DuckDBPyConnection) -> None:
        n = load_advanced_stats(conn, 2024, pd.DataFrame(), pd.DataFrame())
        assert n == 0
