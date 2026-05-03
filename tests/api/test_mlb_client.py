"""
tests/api/test_mlb_client.py

Unit tests for src/api/mlb_client.py.

ALL external calls (HTTP) are mocked — no real network calls are made in any test.
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
    get_active_mlb_players,
    get_batter_stats,
    get_daily_game_schedule,
    get_minor_league_stats,
    get_pitcher_stats,
    get_player_info,
    get_recent_callups,
    get_savant_batter_advanced,
    get_savant_pitcher_advanced,
    get_season_stats_for_projections,
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


# ── Sample boxscore data for batter/pitcher stat tests ───────────────────────

_SAMPLE_BOXSCORE: dict[str, Any] = {
    "teams": {
        "home": {
            "players": {
                "ID660271": {
                    "person": {"id": 660271},
                    "stats": {
                        "batting": {
                            "atBats": 4,
                            "hits": 2,
                            "homeRuns": 1,
                            "doubles": 0,
                            "triples": 0,
                            "stolenBases": 1,
                            "baseOnBalls": 1,
                            "hitByPitch": 0,
                            "sacFlies": 0,
                            "plateAppearances": 5,
                        },
                        "fielding": {"errors": 0, "chances": 3},
                    },
                },
                "ID543000": {
                    "person": {"id": 543000},
                    "stats": {
                        "pitching": {
                            "inningsPitched": "7.0",
                            "wins": 1,
                            "strikeOuts": 8,
                            "baseOnBalls": 2,
                            "hits": 5,
                            "saves": 0,
                            "holds": 0,
                        },
                    },
                },
            },
        },
        "away": {
            "players": {
                "ID665742": {
                    "person": {"id": 665742},
                    "stats": {
                        "pitching": {
                            "inningsPitched": "1.0",
                            "wins": 0,
                            "strikeOuts": 2,
                            "baseOnBalls": 0,
                            "hits": 1,
                            "saves": 1,
                            "holds": 0,
                        },
                    },
                },
            },
        },
    },
}

_SAMPLE_SCHEDULE_FOR_BOXSCORE: dict[str, Any] = {
    "dates": [
        {
            "games": [
                {
                    "gamePk": 746001,
                    "status": {"abstractGameState": "Final"},
                    "teams": {
                        "home": {"team": {"id": 147}},
                        "away": {"team": {"id": 111}},
                    },
                }
            ]
        }
    ]
}


def _mock_requests_for_boxscore(
    url: str, params: dict[str, Any] | None = None, timeout: int = 30
) -> MagicMock:
    """Mock requests.get for schedule + boxscore calls."""
    if "schedule" in url:
        return _make_response(_SAMPLE_SCHEDULE_FOR_BOXSCORE)
    if "boxscore" in url:
        return _make_response(_SAMPLE_BOXSCORE)
    return _make_response({})


# ── get_batter_stats tests ────────────────────────────────────────────────────


class TestGetBatterStats:
    def test_returns_empty_df_when_no_games(self) -> None:
        """Returns empty DataFrame with correct columns when no games found."""
        with patch("requests.get", return_value=_make_response({"dates": []})):
            df = get_batter_stats(datetime.date(2026, 4, 1), datetime.date(2026, 4, 7))

        assert list(df.columns) == [
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
        """Returns DataFrame with fact_player_stats_daily batter columns."""
        with patch("requests.get", side_effect=_mock_requests_for_boxscore):
            df = get_batter_stats(datetime.date(2026, 4, 7), datetime.date(2026, 4, 7))

        expected_cols = [
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
        end_date = datetime.date(2026, 4, 7)
        with patch("requests.get", side_effect=_mock_requests_for_boxscore):
            df = get_batter_stats(datetime.date(2026, 4, 7), end_date)

        assert not df.empty
        assert all(df["stat_date"] == end_date)

    def test_mlb_id_set_from_boxscore(self) -> None:
        """mlb_id is set from boxscore player data."""
        with patch("requests.get", side_effect=_mock_requests_for_boxscore):
            df = get_batter_stats(datetime.date(2026, 4, 7), datetime.date(2026, 4, 7))

        assert 660271 in df["mlb_id"].values

    def test_total_bases_computed(self) -> None:
        """TB = singles + 2*doubles + 3*triples + 4*HR."""
        with patch("requests.get", side_effect=_mock_requests_for_boxscore):
            df = get_batter_stats(datetime.date(2026, 4, 7), datetime.date(2026, 4, 7))

        row = df[df["mlb_id"] == 660271].iloc[0]
        # 2 hits, 1 HR, 0 doubles, 0 triples → 1 single + 4*1 HR = 5 TB
        assert row["tb"] == 5


# ── get_pitcher_stats tests ───────────────────────────────────────────────────


class TestGetPitcherStats:
    def test_returns_empty_df_when_no_games(self) -> None:
        """Returns empty DataFrame with correct columns when no games found."""
        with patch("requests.get", return_value=_make_response({"dates": []})):
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
        """Returns DataFrame with pitcher stat columns."""
        with patch("requests.get", side_effect=_mock_requests_for_boxscore):
            df = get_pitcher_stats(datetime.date(2026, 4, 7), datetime.date(2026, 4, 7))

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
        with patch("requests.get", side_effect=_mock_requests_for_boxscore):
            df = get_pitcher_stats(datetime.date(2026, 4, 7), datetime.date(2026, 4, 7))

        sp_row = df[df["mlb_id"] == 543000].iloc[0]
        # 8 K / 2 BB = 4.0
        assert pytest.approx(sp_row["k_bb"], rel=1e-3) == 4.0

    def test_sv_h_computed(self) -> None:
        """sv_h = sv + holds."""
        with patch("requests.get", side_effect=_mock_requests_for_boxscore):
            df = get_pitcher_stats(datetime.date(2026, 4, 7), datetime.date(2026, 4, 7))

        closer_row = df[df["mlb_id"] == 665742].iloc[0]
        assert closer_row["sv_h"] == 1  # 1 save + 0 holds

    def test_whip_computed(self) -> None:
        """WHIP = (walks_allowed + hits_allowed) / IP."""
        with patch("requests.get", side_effect=_mock_requests_for_boxscore):
            df = get_pitcher_stats(datetime.date(2026, 4, 7), datetime.date(2026, 4, 7))

        sp_row = df[df["mlb_id"] == 543000].iloc[0]
        # (2 + 5) / 7.0 = 1.0
        assert pytest.approx(sp_row["whip"], rel=1e-3) == 1.0


# ── get_steamer_projections tests ─────────────────────────────────────────────


class TestGetSteamerProjections:
    def test_returns_empty_df_with_correct_columns(self) -> None:
        """Returns empty DataFrame with correct columns (now a stub)."""
        df = get_steamer_projections(2026)

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
            "games_per_day",
            "games_remaining",
            "source",
        ]
        assert list(df.columns) == expected_cols

    def test_does_not_raise(self) -> None:
        """Never raises — always returns a valid DataFrame."""
        try:
            df = get_steamer_projections(2026)
        except Exception as exc:
            pytest.fail(f"get_steamer_projections raised unexpectedly: {exc}")

        assert isinstance(df, pd.DataFrame)


# ── get_season_stats_for_projections tests ────────────────────────────────────


class TestGetSeasonStatsForProjections:
    def _season_hitting_response(self) -> dict[str, Any]:
        return {
            "stats": [
                {
                    "group": {"displayName": "hitting"},
                    "splits": [
                        {
                            "stat": {
                                "gamesPlayed": 20,
                                "atBats": 80,
                                "hits": 24,
                                "homeRuns": 5,
                                "stolenBases": 3,
                                "baseOnBalls": 10,
                                "totalBases": 45,
                                "avg": ".300",
                                "ops": ".900",
                                "fielding": ".980",
                            }
                        }
                    ],
                }
            ]
        }

    def _season_pitching_response(self) -> dict[str, Any]:
        return {
            "stats": [
                {
                    "group": {"displayName": "pitching"},
                    "splits": [
                        {
                            "stat": {
                                "gamesPlayed": 5,
                                "inningsPitched": "30.0",
                                "wins": 3,
                                "strikeOuts": 35,
                                "baseOnBalls": 8,
                                "hits": 25,
                                "saves": 0,
                                "holds": 0,
                                "whip": "1.10",
                            }
                        }
                    ],
                }
            ]
        }

    def test_returns_empty_when_no_players(self) -> None:
        """Returns empty DataFrame when mlb_ids list is empty."""
        df = get_season_stats_for_projections([], 2026)
        assert df.empty

    def test_returns_correct_columns(self) -> None:
        """Returns DataFrame with all projection columns."""
        with patch(
            "requests.get",
            return_value=_make_response(self._season_hitting_response()),
        ):
            df = get_season_stats_for_projections([660271], 2026)

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
            "games_per_day",
            "games_remaining",
            "source",
        ]
        assert list(df.columns) == expected_cols

    def test_source_is_mlb_pace(self) -> None:
        """Source column is 'mlb_pace'."""
        with patch(
            "requests.get",
            return_value=_make_response(self._season_hitting_response()),
        ):
            df = get_season_stats_for_projections([660271], 2026)

        assert not df.empty
        assert (df["source"] == "mlb_pace").all()

    def test_stats_scaled_to_per_game(self) -> None:
        """Counting stats are divided by games_played to give per-game rates."""
        with patch(
            "requests.get",
            return_value=_make_response(self._season_hitting_response()),
        ):
            df = get_season_stats_for_projections([660271], 2026)

        row = df.iloc[0]
        # 24 hits in 20 games = 1.2 per game
        assert pytest.approx(float(row["proj_h"]), rel=1e-2) == 1.2

    def test_handles_api_failure_gracefully(self) -> None:
        """Returns empty DataFrame when API fails."""
        with patch(
            "requests.get",
            side_effect=requests.RequestException("timeout"),
        ):
            df = get_season_stats_for_projections([660271], 2026)

        assert df.empty


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


# ── get_active_mlb_players ────────────────────────────────────────────────────


class TestGetActiveMlbPlayers:
    """Tests for get_active_mlb_players()."""

    @patch("src.api.mlb_client._mlb_get")
    def test_returns_active_players(self, mock_get: MagicMock) -> None:
        """Returns DataFrame with full_name and mlb_id columns."""
        mock_get.return_value = {
            "people": [
                {
                    "id": 545361,
                    "fullFMLName": "Michael Nelson Trout",
                    "fullName": "Mike Trout",
                },
                {
                    "id": 660271,
                    "fullFMLName": "Shohei Ohtani",
                    "fullName": "Shohei Ohtani",
                },
            ]
        }
        df = get_active_mlb_players(2026)

        assert len(df) == 2
        assert list(df.columns) == ["full_name", "mlb_id"]
        assert df.iloc[0]["mlb_id"] == 545361
        assert df.iloc[0]["full_name"] == "Mike Trout"

    @patch("src.api.mlb_client._mlb_get")
    def test_skips_entries_without_id(self, mock_get: MagicMock) -> None:
        """Skips people entries missing an id."""
        mock_get.return_value = {
            "people": [
                {"id": 545361, "fullName": "Mike Trout"},
                {"fullName": "Missing ID"},
            ]
        }
        df = get_active_mlb_players(2026)
        assert len(df) == 1

    @patch("src.api.mlb_client._mlb_get")
    def test_returns_empty_on_api_error(self, mock_get: MagicMock) -> None:
        """Returns empty DataFrame on request failure."""
        mock_get.side_effect = requests.RequestException("timeout")
        df = get_active_mlb_players(2026)
        assert df.empty
        assert list(df.columns) == ["full_name", "mlb_id"]


# ── _empty_df helper test ─────────────────────────────────────────────────────


def test_empty_df_has_correct_columns() -> None:
    """_empty_df returns an empty DataFrame with the given columns."""
    cols = ["a", "b", "c"]
    df = _empty_df(cols)
    assert df.empty
    assert list(df.columns) == cols


# ── Savant advanced stats ─────────────────────────────────────────────────────


class TestGetSavantBatterAdvanced:
    """Tests for get_savant_batter_advanced (pybaseball is mocked)."""

    def test_merges_three_sources_on_player_id(self) -> None:
        """xwOBA, barrels, and percentiles join on player_id into one frame."""
        fake_expected = pd.DataFrame(
            [{"player_id": 1, "est_woba": 0.380}, {"player_id": 2, "est_woba": 0.290}]
        )
        fake_barrels = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "brl_percent": 10.5,
                    "ev95percent": 45.2,
                    "avg_hit_angle": 12.3,
                    "anglesweetspotpercent": 35.0,
                }
            ]
        )
        fake_pctile = pd.DataFrame(
            [
                {"player_id": 1, "bat_speed": 75.0, "sprint_speed": 50.0},
                {"player_id": 2, "bat_speed": 40.0, "sprint_speed": 60.0},
            ]
        )
        with (
            patch("pybaseball.statcast_batter_expected_stats") as m1,
            patch("pybaseball.statcast_batter_exitvelo_barrels") as m2,
            patch("pybaseball.statcast_batter_percentile_ranks") as m3,
        ):
            m1.return_value = fake_expected
            m2.return_value = fake_barrels
            m3.return_value = fake_pctile
            out = get_savant_batter_advanced(2024)

        assert not out.empty
        assert set(out["mlb_id"]) == {1, 2}
        p1 = out[out["mlb_id"] == 1].iloc[0]
        assert p1["xwoba"] == pytest.approx(0.380)
        assert p1["barrel_pct"] == pytest.approx(10.5)
        assert p1["hard_hit_pct"] == pytest.approx(45.2)
        assert p1["bat_speed_pctile"] == pytest.approx(75.0)
        # Player 2 has no barrel data — barrel_pct should be NaN
        p2 = out[out["mlb_id"] == 2].iloc[0]
        assert pd.isna(p2["barrel_pct"])
        assert p2["bat_speed_pctile"] == pytest.approx(40.0)

    def test_returns_empty_when_all_sources_fail(self) -> None:
        """If every Savant call errors, return an empty frame with correct cols."""
        with (
            patch("pybaseball.statcast_batter_expected_stats") as m1,
            patch("pybaseball.statcast_batter_exitvelo_barrels") as m2,
            patch("pybaseball.statcast_batter_percentile_ranks") as m3,
        ):
            m1.side_effect = RuntimeError("down")
            m2.side_effect = RuntimeError("down")
            m3.side_effect = RuntimeError("down")
            out = get_savant_batter_advanced(2024)
        assert out.empty
        assert "xwoba" in out.columns


class TestGetSavantPitcherAdvanced:
    def test_merges_expected_and_barrels(self) -> None:
        fake_expected = pd.DataFrame(
            [
                {"player_id": 1, "xera": 3.50, "est_woba": 0.290},
                {"player_id": 2, "xera": 4.20, "est_woba": 0.330},
            ]
        )
        fake_barrels = pd.DataFrame(
            [{"player_id": 1, "brl_percent": 6.5, "ev95percent": 32.0}]
        )
        with (
            patch("pybaseball.statcast_pitcher_expected_stats") as m1,
            patch("pybaseball.statcast_pitcher_exitvelo_barrels") as m2,
        ):
            m1.return_value = fake_expected
            m2.return_value = fake_barrels
            out = get_savant_pitcher_advanced(2024)

        assert set(out["mlb_id"]) == {1, 2}
        p1 = out[out["mlb_id"] == 1].iloc[0]
        assert p1["xera"] == pytest.approx(3.50)
        assert p1["xwoba_against"] == pytest.approx(0.290)
        assert p1["barrel_pct_against"] == pytest.approx(6.5)

    def test_returns_empty_when_barrels_fail(self) -> None:
        """Barrels endpoint failure should not prevent xERA/xwOBA from returning."""
        fake_expected = pd.DataFrame(
            [{"player_id": 1, "xera": 3.50, "est_woba": 0.290}]
        )
        with (
            patch("pybaseball.statcast_pitcher_expected_stats") as m1,
            patch("pybaseball.statcast_pitcher_exitvelo_barrels") as m2,
        ):
            m1.return_value = fake_expected
            m2.side_effect = RuntimeError("parse error")
            out = get_savant_pitcher_advanced(2024)
        assert len(out) == 1
        assert out.iloc[0]["xera"] == pytest.approx(3.50)
        assert pd.isna(out.iloc[0]["barrel_pct_against"])
