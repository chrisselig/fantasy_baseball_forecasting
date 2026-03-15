"""
tests/api/test_mlb_client.py

Unit tests for src/api/mlb_client.py.

ALL external calls (HTTP + pybaseball) are mocked — no real network or
pybaseball calls are made in any test.
"""

from __future__ import annotations

import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from src.api.mlb_client import (
    _empty_df,
    build_player_id_crosswalk,
    get_batter_stats,
    get_daily_game_schedule,
    get_minor_league_stats,
    get_pitcher_stats,
    get_player_info,
    get_recent_callups,
    get_steamer_projections,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_response(json_data: dict[str, Any], status_code: int = 200) -> MagicMock:
    """Create a mock requests.Response object."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    if status_code >= 400:
        mock.raise_for_status.side_effect = requests.HTTPError(
            f"HTTP {status_code}", response=mock
        )
    else:
        mock.raise_for_status.return_value = None
    return mock


_SAMPLE_TRANSACTIONS = {
    "transactions": [
        {
            "typeCode": "CU",
            "person": {"id": 12345, "fullName": "John Doe"},
            "toTeam": {"abbreviation": "NYY"},
            "fromOrg": {"name": "Triple-A East"},
            "date": "2026-03-10",
        },
        {
            "typeCode": "DFA",  # should be filtered out — not a call-up
            "person": {"id": 99999, "fullName": "Jane Smith"},
            "toTeam": {"abbreviation": "BOS"},
            "fromOrg": {"name": "Triple-A East"},
            "date": "2026-03-10",
        },
        {
            "typeCode": "CU",
            "person": {"id": 67890, "fullName": "Alex Rodriguez Jr"},
            "toTeam": {"abbreviation": "LAD"},
            "fromOrg": {"name": "Double-A South"},
            "date": "2026-03-11",
        },
    ]
}

_SAMPLE_PLAYER = {
    "people": [
        {
            "id": 12345,
            "fullName": "John Doe",
            "currentTeam": {"abbreviation": "NYY"},
            "primaryPosition": {"abbreviation": "SS"},
            "batSide": {"code": "R"},
            "pitchHand": {"code": "R"},
            "active": True,
        }
    ]
}

_SAMPLE_SCHEDULE = {
    "dates": [
        {
            "games": [
                {
                    "officialDate": "2026-03-14",
                    "teams": {
                        "home": {
                            "team": {"id": 147, "abbreviation": "NYY"},
                            "probablePitcher": {"fullName": "Gerrit Cole"},
                        },
                        "away": {
                            "team": {"id": 111, "abbreviation": "BOS"},
                            "probablePitcher": None,
                        },
                    },
                }
            ]
        }
    ]
}


# ── get_recent_callups tests ──────────────────────────────────────────────────


class TestGetRecentCallups:
    def test_returns_correct_columns(self) -> None:
        """Returns DataFrame with expected columns."""
        with patch("requests.get", return_value=_make_response(_SAMPLE_TRANSACTIONS)):
            df = get_recent_callups(days=7)

        assert list(df.columns) == [
            "mlb_id",
            "full_name",
            "team",
            "transaction_date",
            "from_level",
        ]

    def test_filters_to_callups_only(self) -> None:
        """Only typeCode='CU' transactions are returned."""
        with patch("requests.get", return_value=_make_response(_SAMPLE_TRANSACTIONS)):
            df = get_recent_callups(days=7)

        # DFA transaction (99999) should not appear
        assert 99999 not in df["mlb_id"].values
        assert len(df) == 2

    def test_correct_values_parsed(self) -> None:
        """mlb_id, full_name, team, from_level parsed correctly."""
        with patch("requests.get", return_value=_make_response(_SAMPLE_TRANSACTIONS)):
            df = get_recent_callups(days=7)

        first = df[df["mlb_id"] == 12345].iloc[0]
        assert first["full_name"] == "John Doe"
        assert first["team"] == "NYY"
        assert first["from_level"] == "Triple-A East"
        assert first["transaction_date"] == datetime.date(2026, 3, 10)

    def test_empty_transactions(self) -> None:
        """Returns empty DataFrame with correct columns when no transactions."""
        with patch("requests.get", return_value=_make_response({"transactions": []})):
            df = get_recent_callups(days=7)

        assert df.empty
        assert list(df.columns) == [
            "mlb_id",
            "full_name",
            "team",
            "transaction_date",
            "from_level",
        ]

    def test_no_callup_transactions(self) -> None:
        """Returns empty DataFrame when all transactions are non-CU types."""
        non_cu_data: dict[str, Any] = {
            "transactions": [
                {
                    "typeCode": "DFA",
                    "person": {"id": 11111, "fullName": "Player X"},
                    "toTeam": {"abbreviation": "ATL"},
                    "fromOrg": {"name": "AAA"},
                    "date": "2026-03-10",
                }
            ]
        }
        with patch("requests.get", return_value=_make_response(non_cu_data)):
            df = get_recent_callups(days=7)

        assert df.empty


# ── get_player_info tests ─────────────────────────────────────────────────────


class TestGetPlayerInfo:
    def test_returns_correct_keys(self) -> None:
        """Returns dict with all expected keys."""
        with patch("requests.get", return_value=_make_response(_SAMPLE_PLAYER)):
            result = get_player_info(12345)

        expected_keys = {
            "mlb_id",
            "full_name",
            "team",
            "positions",
            "bats",
            "throws",
            "status",
        }
        assert set(result.keys()) == expected_keys

    def test_correct_values(self) -> None:
        """Parses player metadata correctly."""
        with patch("requests.get", return_value=_make_response(_SAMPLE_PLAYER)):
            result = get_player_info(12345)

        assert result["mlb_id"] == 12345
        assert result["full_name"] == "John Doe"
        assert result["team"] == "NYY"
        assert result["positions"] == ["SS"]
        assert result["bats"] == "R"
        assert result["throws"] == "R"
        assert result["status"] == "Active"

    def test_empty_people_returns_defaults(self) -> None:
        """Returns default dict when API returns empty people list."""
        with patch("requests.get", return_value=_make_response({"people": []})):
            result = get_player_info(99999)

        assert result["mlb_id"] == 99999
        assert result["full_name"] == ""
        assert result["positions"] == []

    def test_inactive_player_status(self) -> None:
        """Sets status to 'Inactive' for inactive players."""
        inactive_player = {
            "people": [
                {
                    "id": 55555,
                    "fullName": "Injured Player",
                    "currentTeam": {"abbreviation": "CHC"},
                    "primaryPosition": {"abbreviation": "OF"},
                    "batSide": {"code": "L"},
                    "pitchHand": {"code": "R"},
                    "active": False,
                }
            ]
        }
        with patch("requests.get", return_value=_make_response(inactive_player)):
            result = get_player_info(55555)

        assert result["status"] == "Inactive"


# ── get_daily_game_schedule tests ─────────────────────────────────────────────


class TestGetDailyGameSchedule:
    def test_returns_correct_columns(self) -> None:
        """Returns DataFrame with expected columns."""
        with patch("requests.get", return_value=_make_response(_SAMPLE_SCHEDULE)):
            df = get_daily_game_schedule(datetime.date(2026, 3, 14))

        assert list(df.columns) == [
            "mlb_id",
            "game_date",
            "opponent_team",
            "home_away",
            "probable_pitcher",
        ]

    def test_both_teams_returned(self) -> None:
        """Both home and away teams appear as separate rows."""
        with patch("requests.get", return_value=_make_response(_SAMPLE_SCHEDULE)):
            df = get_daily_game_schedule(datetime.date(2026, 3, 14))

        assert len(df) == 2
        assert set(df["home_away"].tolist()) == {"home", "away"}

    def test_home_team_opponent_is_away(self) -> None:
        """Home team's opponent_team is the away team abbreviation."""
        with patch("requests.get", return_value=_make_response(_SAMPLE_SCHEDULE)):
            df = get_daily_game_schedule(datetime.date(2026, 3, 14))

        home_row = df[df["home_away"] == "home"].iloc[0]
        assert home_row["mlb_id"] == 147
        assert home_row["opponent_team"] == "BOS"
        assert home_row["probable_pitcher"] == "Gerrit Cole"

    def test_away_team_no_pitcher(self) -> None:
        """probable_pitcher is None when no pitcher listed."""
        with patch("requests.get", return_value=_make_response(_SAMPLE_SCHEDULE)):
            df = get_daily_game_schedule(datetime.date(2026, 3, 14))

        away_row = df[df["home_away"] == "away"].iloc[0]
        assert pd.isna(away_row["probable_pitcher"])

    def test_empty_schedule(self) -> None:
        """Returns empty DataFrame with correct columns when no games."""
        with patch("requests.get", return_value=_make_response({"dates": []})):
            df = get_daily_game_schedule(datetime.date(2026, 3, 14))

        assert df.empty
        assert list(df.columns) == [
            "mlb_id",
            "game_date",
            "opponent_team",
            "home_away",
            "probable_pitcher",
        ]


# ── get_minor_league_stats tests ──────────────────────────────────────────────


class TestGetMinorLeagueStats:
    def _aaa_response(self) -> dict[str, Any]:
        return {
            "stats": [
                {
                    "splits": [
                        {
                            "stat": {
                                "atBats": 200,
                                "hits": 60,
                                "homeRuns": 10,
                                "stolenBases": 5,
                                "baseOnBalls": 25,
                                "avg": ".300",
                                "ops": ".850",
                                "inningsPitched": None,
                                "strikeOuts": None,
                                "whip": None,
                                "era": None,
                            }
                        }
                    ]
                }
            ]
        }

    def _empty_response(self) -> dict[str, Any]:
        return {"stats": []}

    def test_returns_aaa_when_data_available(self) -> None:
        """Returns AAA stats when AAA data is present."""
        with patch("requests.get", return_value=_make_response(self._aaa_response())):
            df = get_minor_league_stats(12345, 2026)

        assert not df.empty
        assert df.iloc[0]["level"] == "AAA"

    def test_falls_back_to_aa_when_aaa_empty(self) -> None:
        """Falls back to AA when AAA returns no data."""
        aa_response: dict[str, Any] = {
            "stats": [
                {
                    "splits": [
                        {
                            "stat": {
                                "atBats": 150,
                                "hits": 45,
                                "homeRuns": 7,
                                "stolenBases": 3,
                                "baseOnBalls": 18,
                                "avg": ".300",
                                "ops": ".810",
                                "inningsPitched": None,
                                "strikeOuts": None,
                                "whip": None,
                                "era": None,
                            }
                        }
                    ]
                }
            ]
        }

        def side_effect(
            url: str, params: dict[str, Any] | None = None, timeout: int = 30
        ) -> MagicMock:
            sport_id = (params or {}).get("sportId")
            if sport_id == 11:  # AAA — return empty
                return _make_response(self._empty_response())
            if sport_id == 12:  # AA — return data
                return _make_response(aa_response)
            return _make_response(self._empty_response())

        with patch("requests.get", side_effect=side_effect):
            df = get_minor_league_stats(12345, 2026)

        assert not df.empty
        assert df.iloc[0]["level"] == "AA"

    def test_returns_empty_when_no_data_at_any_level(self) -> None:
        """Returns empty DataFrame with correct columns when no MiLB data found."""
        with patch("requests.get", return_value=_make_response(self._empty_response())):
            df = get_minor_league_stats(12345, 2026)

        assert df.empty
        assert "mlb_id" in df.columns
        assert "level" in df.columns

    def test_correct_columns(self) -> None:
        """Returns DataFrame with expected columns."""
        with patch("requests.get", return_value=_make_response(self._aaa_response())):
            df = get_minor_league_stats(12345, 2026)

        expected_cols = {
            "mlb_id",
            "season",
            "level",
            "ab",
            "h",
            "hr",
            "sb",
            "bb",
            "avg",
            "ops",
            "ip",
            "k",
            "whip",
            "era",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_mlb_id_set_correctly(self) -> None:
        """mlb_id in result matches the input mlb_id."""
        with patch("requests.get", return_value=_make_response(self._aaa_response())):
            df = get_minor_league_stats(12345, 2026)

        assert df.iloc[0]["mlb_id"] == 12345


# ── get_batter_stats tests ────────────────────────────────────────────────────


class TestGetBatterStats:
    def test_returns_empty_df_when_pybaseball_fails(self) -> None:
        """Returns empty DataFrame with correct columns when pybaseball fails."""
        import sys
        import types

        broken = types.ModuleType("pybaseball")
        broken.batting_stats = MagicMock(side_effect=Exception("Connection error"))  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"pybaseball": broken}):
            df = get_batter_stats(datetime.date(2026, 4, 1), datetime.date(2026, 4, 7))

        # Should return empty with correct columns
        assert list(df.columns) == [
            "player_id",
            "mlb_id",
            "stat_date",
            "ab",
            "h",
            "hr",
            "sb",
            "bb",
            "hbp",
            "sf",
            "tb",
            "avg",
            "ops",
            "fpct",
            "errors",
            "chances",
        ]
        assert df.empty

    def test_returns_correct_columns_on_success(self) -> None:
        """Returns DataFrame with fact_player_stats_daily batter columns on success."""
        mock_df = pd.DataFrame(
            {
                "IDfg": [12345],
                "AB": [400],
                "H": [120],
                "HR": [20],
                "SB": [10],
                "BB": [50],
                "HBP": [5],
                "SF": [3],
                "AVG": [0.300],
                "OPS": [0.850],
            }
        )

        import sys

        mock_pybaseball = MagicMock()
        mock_pybaseball.batting_stats.return_value = mock_df

        with patch.dict(sys.modules, {"pybaseball": mock_pybaseball}):
            df = get_batter_stats(datetime.date(2026, 4, 1), datetime.date(2026, 4, 7))

        expected_cols = [
            "player_id",
            "mlb_id",
            "stat_date",
            "ab",
            "h",
            "hr",
            "sb",
            "bb",
            "hbp",
            "sf",
            "tb",
            "avg",
            "ops",
            "fpct",
            "errors",
            "chances",
        ]
        assert list(df.columns) == expected_cols

    def test_stat_date_set_to_end_date(self) -> None:
        """stat_date column is set to end_date."""
        mock_df = pd.DataFrame(
            {
                "IDfg": [12345],
                "AB": [400],
                "H": [120],
                "HR": [20],
                "SB": [10],
                "BB": [50],
                "HBP": [5],
                "SF": [3],
                "AVG": [0.300],
                "OPS": [0.850],
            }
        )

        import sys

        mock_pybaseball = MagicMock()
        mock_pybaseball.batting_stats.return_value = mock_df

        end_date = datetime.date(2026, 4, 7)
        with patch.dict(sys.modules, {"pybaseball": mock_pybaseball}):
            df = get_batter_stats(datetime.date(2026, 4, 1), end_date)

        assert all(df["stat_date"] == end_date)


def _patched_import_batting_fail(name: str, *args: Any, **kwargs: Any) -> Any:
    """Simulate pybaseball import that raises on batting_stats call."""
    if name == "pybaseball":
        mock = MagicMock()
        mock.batting_stats.side_effect = Exception("pybaseball failure")
        return mock
    import builtins

    return builtins.__import__(name, *args, **kwargs)


# ── get_pitcher_stats tests ───────────────────────────────────────────────────


class TestGetPitcherStats:
    def test_returns_empty_df_when_pybaseball_fails(self) -> None:
        """Returns empty DataFrame with correct columns when pybaseball fails."""
        import sys

        mock_pybaseball = MagicMock()
        mock_pybaseball.pitching_stats.side_effect = Exception("Network error")

        with patch.dict(sys.modules, {"pybaseball": mock_pybaseball}):
            df = get_pitcher_stats(datetime.date(2026, 4, 1), datetime.date(2026, 4, 7))

        assert df.empty
        expected_cols = [
            "mlb_id",
            "stat_date",
            "ip",
            "w",
            "k",
            "walks_allowed",
            "hits_allowed",
            "sv",
            "holds",
            "whip",
            "k_bb",
            "sv_h",
        ]
        assert list(df.columns) == expected_cols

    def test_returns_correct_columns_on_success(self) -> None:
        """Returns DataFrame with pitcher stat columns on success."""
        mock_df = pd.DataFrame(
            {
                "IP": [180.0],
                "W": [15],
                "SO": [200],
                "BB": [50],
                "H": [160],
                "SV": [0],
                "HLD": [0],
                "WHIP": [1.17],
            }
        )

        import sys

        mock_pybaseball = MagicMock()
        mock_pybaseball.pitching_stats.return_value = mock_df

        with patch.dict(sys.modules, {"pybaseball": mock_pybaseball}):
            df = get_pitcher_stats(datetime.date(2026, 4, 1), datetime.date(2026, 4, 7))

        expected_cols = [
            "mlb_id",
            "stat_date",
            "ip",
            "w",
            "k",
            "walks_allowed",
            "hits_allowed",
            "sv",
            "holds",
            "whip",
            "k_bb",
            "sv_h",
        ]
        assert list(df.columns) == expected_cols

    def test_k_bb_computed(self) -> None:
        """k_bb is computed from k/walks_allowed."""
        mock_df = pd.DataFrame(
            {
                "IP": [100.0],
                "W": [8],
                "SO": [100],
                "BB": [25],
                "H": [90],
                "SV": [0],
                "HLD": [5],
                "WHIP": [1.15],
            }
        )

        import sys

        mock_pybaseball = MagicMock()
        mock_pybaseball.pitching_stats.return_value = mock_df

        with patch.dict(sys.modules, {"pybaseball": mock_pybaseball}):
            df = get_pitcher_stats(datetime.date(2026, 4, 1), datetime.date(2026, 4, 7))

        assert pytest.approx(df.iloc[0]["k_bb"], rel=1e-3) == 4.0

    def test_sv_h_computed(self) -> None:
        """sv_h = sv + holds."""
        mock_df = pd.DataFrame(
            {
                "IP": [60.0],
                "W": [4],
                "SO": [70],
                "BB": [20],
                "H": [55],
                "SV": [10],
                "HLD": [15],
                "WHIP": [1.25],
            }
        )

        import sys

        mock_pybaseball = MagicMock()
        mock_pybaseball.pitching_stats.return_value = mock_df

        with patch.dict(sys.modules, {"pybaseball": mock_pybaseball}):
            df = get_pitcher_stats(datetime.date(2026, 4, 1), datetime.date(2026, 4, 7))

        assert df.iloc[0]["sv_h"] == 25


# ── get_steamer_projections tests ─────────────────────────────────────────────


class TestGetSteamerProjections:
    def test_returns_source_unavailable_on_total_failure(self) -> None:
        """Returns empty DataFrame (not raises) with source='unavailable' column on failure."""
        import sys

        mock_pybaseball = MagicMock()
        mock_pybaseball.fg_batting_projections.side_effect = Exception(
            "FanGraphs blocked"
        )
        mock_pybaseball.fg_pitching_projections.side_effect = Exception(
            "FanGraphs blocked"
        )

        with patch.dict(sys.modules, {"pybaseball": mock_pybaseball}):
            df = get_steamer_projections(2026)

        # Should not raise; returns empty DataFrame
        assert df.empty
        expected_cols = [
            "mlb_id",
            "fg_id",
            "proj_h",
            "proj_hr",
            "proj_sb",
            "proj_bb",
            "proj_ab",
            "proj_tb",
            "proj_ip",
            "proj_w",
            "proj_k",
            "proj_walks_allowed",
            "proj_hits_allowed",
            "proj_sv_h",
            "proj_avg",
            "proj_ops",
            "proj_fpct",
            "proj_whip",
            "proj_k_bb",
            "games_remaining",
            "source",
        ]
        assert list(df.columns) == expected_cols

    def test_does_not_raise_on_failure(self) -> None:
        """Never raises — always returns a valid DataFrame."""
        import sys

        mock_pybaseball = MagicMock()
        mock_pybaseball.fg_batting_projections.side_effect = RuntimeError("Boom")
        mock_pybaseball.fg_pitching_projections.side_effect = RuntimeError("Boom")

        with patch.dict(sys.modules, {"pybaseball": mock_pybaseball}):
            try:
                df = get_steamer_projections(2026)
            except Exception as exc:
                pytest.fail(f"get_steamer_projections raised unexpectedly: {exc}")

        assert isinstance(df, pd.DataFrame)

    def test_returns_steamer_source_on_success(self) -> None:
        """Returns source='steamer' rows on successful pull."""
        bat_df = pd.DataFrame(
            {
                "playerid": ["fg123"],
                "H": [150.0],
                "HR": [25.0],
                "SB": [10.0],
                "BB": [55.0],
                "AB": [500.0],
                "AVG": [0.300],
                "OPS": [0.860],
            }
        )

        import sys

        mock_pybaseball = MagicMock()
        mock_pybaseball.fg_batting_projections.return_value = bat_df
        mock_pybaseball.fg_pitching_projections.side_effect = Exception("No pitching")

        with patch.dict(sys.modules, {"pybaseball": mock_pybaseball}):
            df = get_steamer_projections(2026)

        assert not df.empty
        assert (df["source"] == "steamer").all()

    def test_correct_columns_returned(self) -> None:
        """Returns DataFrame with all fact_projections columns."""
        bat_df = pd.DataFrame(
            {
                "playerid": ["fg001", "fg002"],
                "H": [120.0, 80.0],
                "HR": [15.0, 5.0],
                "SB": [8.0, 20.0],
                "BB": [40.0, 30.0],
                "AB": [450.0, 350.0],
                "AVG": [0.267, 0.229],
                "OPS": [0.780, 0.690],
            }
        )

        import sys

        mock_pybaseball = MagicMock()
        mock_pybaseball.fg_batting_projections.return_value = bat_df
        mock_pybaseball.fg_pitching_projections.return_value = pd.DataFrame()

        with patch.dict(sys.modules, {"pybaseball": mock_pybaseball}):
            df = get_steamer_projections(2026)

        expected_cols = [
            "mlb_id",
            "fg_id",
            "proj_h",
            "proj_hr",
            "proj_sb",
            "proj_bb",
            "proj_ab",
            "proj_tb",
            "proj_ip",
            "proj_w",
            "proj_k",
            "proj_walks_allowed",
            "proj_hits_allowed",
            "proj_sv_h",
            "proj_avg",
            "proj_ops",
            "proj_fpct",
            "proj_whip",
            "proj_k_bb",
            "games_remaining",
            "source",
        ]
        assert list(df.columns) == expected_cols


# ── build_player_id_crosswalk tests ──────────────────────────────────────────


class TestBuildPlayerIdCrosswalk:
    def test_returns_correct_columns(self) -> None:
        """Returns DataFrame with full_name, mlb_id, fg_id columns."""
        register_df = pd.DataFrame(
            {
                "key_mlbam": [12345, 67890],
                "key_fangraphs": ["fg001", "fg002"],
                "name_first": ["John", "Alex"],
                "name_last": ["Doe", "Smith"],
            }
        )

        import sys

        mock_pybaseball = MagicMock()
        mock_pybaseball.chadwick_register.return_value = register_df

        with patch.dict(sys.modules, {"pybaseball": mock_pybaseball}):
            df = build_player_id_crosswalk()

        assert list(df.columns) == ["full_name", "mlb_id", "fg_id"]

    def test_full_name_constructed(self) -> None:
        """full_name is correctly constructed from name_first + name_last."""
        register_df = pd.DataFrame(
            {
                "key_mlbam": [12345],
                "key_fangraphs": ["fg001"],
                "name_first": ["John"],
                "name_last": ["Doe"],
            }
        )

        import sys

        mock_pybaseball = MagicMock()
        mock_pybaseball.chadwick_register.return_value = register_df

        with patch.dict(sys.modules, {"pybaseball": mock_pybaseball}):
            df = build_player_id_crosswalk()

        assert df.iloc[0]["full_name"] == "John Doe"

    def test_returns_empty_df_on_failure(self) -> None:
        """Returns empty DataFrame with correct columns when pybaseball fails."""
        import sys

        mock_pybaseball = MagicMock()
        mock_pybaseball.chadwick_register.side_effect = Exception("Network down")

        with patch.dict(sys.modules, {"pybaseball": mock_pybaseball}):
            df = build_player_id_crosswalk()

        assert df.empty
        assert list(df.columns) == ["full_name", "mlb_id", "fg_id"]

    def test_rows_without_mlb_id_excluded(self) -> None:
        """Rows where mlb_id (key_mlbam) is NaN are excluded."""
        register_df = pd.DataFrame(
            {
                "key_mlbam": [12345, None],
                "key_fangraphs": ["fg001", "fg002"],
                "name_first": ["John", "Jane"],
                "name_last": ["Doe", "Smith"],
            }
        )

        import sys

        mock_pybaseball = MagicMock()
        mock_pybaseball.chadwick_register.return_value = register_df

        with patch.dict(sys.modules, {"pybaseball": mock_pybaseball}):
            df = build_player_id_crosswalk()

        assert len(df) == 1
        assert df.iloc[0]["mlb_id"] == 12345


# ── _empty_df helper test ─────────────────────────────────────────────────────


def test_empty_df_has_correct_columns() -> None:
    """_empty_df returns an empty DataFrame with the given columns."""
    cols = ["a", "b", "c"]
    df = _empty_df(cols)
    assert df.empty
    assert list(df.columns) == cols
