"""
tests/api/test_yahoo_client.py

Unit tests for src/api/yahoo_client.py.

All HTTP calls are intercepted by the ``responses`` library — no real network
traffic occurs during these tests.
"""

from __future__ import annotations

import time

import pandas as pd
import pytest
import responses as responses_lib

from src.api.yahoo_client import (
    YahooAPIError,
    YahooAuthError,
    YahooClient,
    _parse_free_agents_response,
    _parse_matchup_response,
    _parse_player_details,
    _parse_roster_response,
    _parse_standings_response,
    _parse_transactions_response,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

CONSUMER_KEY = "test_consumer_key"
CONSUMER_SECRET = "test_consumer_secret"
ACCESS_TOKEN = "test_access_token"
REFRESH_TOKEN = "test_refresh_token"

BASE = "https://fantasysports.yahooapis.com/fantasy/v2/"
TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"


@pytest.fixture()
def client() -> YahooClient:
    """A YahooClient with test credentials and a fresh token timestamp."""
    c = YahooClient(
        consumer_key=CONSUMER_KEY,
        consumer_secret=CONSUMER_SECRET,
        access_token=ACCESS_TOKEN,
        refresh_token=REFRESH_TOKEN,
    )
    # Ensure the token looks fresh so refresh is not triggered in most tests.
    c._token_issued_at = time.time()
    return c


# ── from_env tests ────────────────────────────────────────────────────────────


class TestFromEnv:
    def test_raises_when_all_vars_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for var in [
            "YAHOO_CONSUMER_KEY",
            "YAHOO_CONSUMER_SECRET",
            "YAHOO_ACCESS_TOKEN",
            "YAHOO_REFRESH_TOKEN",
        ]:
            monkeypatch.delenv(var, raising=False)

        with pytest.raises(
            EnvironmentError, match="Missing required environment variables"
        ):
            YahooClient.from_env()

    def test_raises_when_one_var_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YAHOO_CONSUMER_KEY", "k")
        monkeypatch.setenv("YAHOO_CONSUMER_SECRET", "s")
        monkeypatch.setenv("YAHOO_ACCESS_TOKEN", "at")
        monkeypatch.delenv("YAHOO_REFRESH_TOKEN", raising=False)

        with pytest.raises(EnvironmentError, match="YAHOO_REFRESH_TOKEN"):
            YahooClient.from_env()

    def test_succeeds_when_all_vars_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YAHOO_CONSUMER_KEY", "k")
        monkeypatch.setenv("YAHOO_CONSUMER_SECRET", "s")
        monkeypatch.setenv("YAHOO_ACCESS_TOKEN", "at")
        monkeypatch.setenv("YAHOO_REFRESH_TOKEN", "rt")

        c = YahooClient.from_env()
        assert c._consumer_key == "k"
        assert c._consumer_secret == "s"
        assert c._access_token == "at"
        assert c._refresh_token == "rt"


# ── _get error-handling tests ─────────────────────────────────────────────────


class TestGet:
    @responses_lib.activate
    def test_raises_yahoo_auth_error_on_401(self, client: YahooClient) -> None:
        responses_lib.add(
            responses_lib.GET,
            BASE + "some/endpoint",
            status=401,
            body="Unauthorized",
        )
        with pytest.raises(YahooAuthError):
            client._get("some/endpoint")

    @responses_lib.activate
    def test_raises_yahoo_auth_error_on_403(self, client: YahooClient) -> None:
        responses_lib.add(
            responses_lib.GET,
            BASE + "some/endpoint",
            status=403,
            body="Forbidden",
        )
        with pytest.raises(YahooAuthError):
            client._get("some/endpoint")

    @responses_lib.activate
    def test_raises_yahoo_api_error_on_500(self, client: YahooClient) -> None:
        responses_lib.add(
            responses_lib.GET,
            BASE + "some/endpoint",
            status=500,
            body="Internal Server Error",
        )
        with pytest.raises(YahooAPIError):
            client._get("some/endpoint")

    @responses_lib.activate
    def test_returns_json_on_200(self, client: YahooClient) -> None:
        responses_lib.add(
            responses_lib.GET,
            BASE + "some/endpoint",
            status=200,
            json={"fantasy_content": {"ok": True}},
        )
        result = client._get("some/endpoint")
        assert result == {"fantasy_content": {"ok": True}}

    @responses_lib.activate
    def test_adds_format_json_param(self, client: YahooClient) -> None:
        responses_lib.add(
            responses_lib.GET,
            BASE + "some/endpoint",
            status=200,
            json={},
        )
        client._get("some/endpoint")
        assert "format=json" in (responses_lib.calls[0].request.url or "")

    @responses_lib.activate
    def test_bearer_token_in_auth_header(self, client: YahooClient) -> None:
        responses_lib.add(
            responses_lib.GET,
            BASE + "some/endpoint",
            status=200,
            json={},
        )
        client._get("some/endpoint")
        auth_header = responses_lib.calls[0].request.headers.get("Authorization", "")
        assert auth_header == f"Bearer {ACCESS_TOKEN}"


# ── Token refresh tests ───────────────────────────────────────────────────────


class TestTokenRefresh:
    @responses_lib.activate
    def test_refresh_triggered_when_token_expired(self, client: YahooClient) -> None:
        """When the token is old, a refresh call should fire before the GET."""
        # Make the token look ancient (issued 4000 seconds ago = past 1h expiry)
        client._token_issued_at = time.time() - 4000

        responses_lib.add(
            responses_lib.POST,
            TOKEN_URL,
            status=200,
            json={
                "access_token": "new_access_token",
                "refresh_token": "new_refresh_token",
            },
        )
        responses_lib.add(
            responses_lib.GET,
            BASE + "some/endpoint",
            status=200,
            json={"ok": True},
        )

        client._get("some/endpoint")

        # The first call should be the token refresh POST
        assert responses_lib.calls[0].request.url == TOKEN_URL
        assert client._access_token == "new_access_token"
        assert client._refresh_token == "new_refresh_token"

    @responses_lib.activate
    def test_refresh_not_triggered_when_token_fresh(self, client: YahooClient) -> None:
        """When the token is new, no refresh call should be made."""
        client._token_issued_at = time.time()  # brand-new token

        responses_lib.add(
            responses_lib.GET,
            BASE + "some/endpoint",
            status=200,
            json={"ok": True},
        )

        client._get("some/endpoint")
        # Only one call — the actual GET, no POST to token endpoint
        assert len(responses_lib.calls) == 1
        assert "get_token" not in (responses_lib.calls[0].request.url or "")

    @responses_lib.activate
    def test_refresh_raises_yahoo_auth_error_on_401(self, client: YahooClient) -> None:
        client._token_issued_at = time.time() - 4000

        responses_lib.add(
            responses_lib.POST,
            TOKEN_URL,
            status=401,
            body="Unauthorized",
        )

        with pytest.raises(YahooAuthError, match="Token refresh failed"):
            client._refresh_token_if_needed()


# ── get_my_roster tests ───────────────────────────────────────────────────────


def _roster_json(team_key: str = "423.l.87941.t.1") -> dict:  # type: ignore[type-arg]
    """Minimal realistic Yahoo roster JSON for a single team."""
    return {
        "fantasy_content": {
            "team": [
                [{"team_key": team_key}, {"name": "Test Team"}],
                {
                    "roster": {
                        "0": {
                            "players": {
                                "0": {
                                    "player": [
                                        [
                                            {"player_key": "423.p.7578"},
                                            {"name": {"full": "Yordan Alvarez"}},
                                            {"acquisition_type": "draft"},
                                        ],
                                        {"selected_position": [{"position": "1B"}]},
                                    ]
                                },
                                "count": 1,
                            }
                        }
                    }
                },
            ]
        }
    }


class TestGetMyRoster:
    @responses_lib.activate
    def test_returns_dataframe_with_correct_columns(
        self, client: YahooClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("YAHOO_LEAGUE_ID", "87941")
        monkeypatch.setenv("YAHOO_TEAM_ID", "1")

        responses_lib.add(
            responses_lib.GET,
            BASE + "team/423.l.87941.t.1/roster",
            status=200,
            json=_roster_json(),
        )

        df = client.get_my_roster(week=1)
        assert isinstance(df, pd.DataFrame)
        required = {
            "team_id",
            "player_id",
            "snapshot_date",
            "roster_slot",
            "acquisition_type",
        }
        assert required.issubset(set(df.columns))

    @responses_lib.activate
    def test_roster_contains_expected_player(
        self, client: YahooClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("YAHOO_LEAGUE_ID", "87941")
        monkeypatch.setenv("YAHOO_TEAM_ID", "1")

        responses_lib.add(
            responses_lib.GET,
            BASE + "team/423.l.87941.t.1/roster",
            status=200,
            json=_roster_json(),
        )

        df = client.get_my_roster(week=1)
        assert "423.p.7578" in df["player_id"].values
        assert df.loc[df["player_id"] == "423.p.7578", "roster_slot"].iloc[0] == "1B"


# ── get_all_rosters tests ─────────────────────────────────────────────────────


def _all_rosters_json() -> dict:  # type: ignore[type-arg]
    team_payload = _roster_json("423.l.87941.t.1")["fantasy_content"]["team"]
    return {
        "fantasy_content": {
            "league": [
                [{"league_id": "87941"}],
                {
                    "teams": {
                        "0": {"team": team_payload},
                        "count": 1,
                    }
                },
            ]
        }
    }


class TestGetAllRosters:
    @responses_lib.activate
    def test_returns_dataframe_with_correct_columns(
        self, client: YahooClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("YAHOO_LEAGUE_ID", "87941")

        responses_lib.add(
            responses_lib.GET,
            BASE + "league/423.l.87941/teams/roster",
            status=200,
            json=_all_rosters_json(),
        )

        df = client.get_all_rosters(week=1)
        assert isinstance(df, pd.DataFrame)
        required = {
            "team_id",
            "player_id",
            "snapshot_date",
            "roster_slot",
            "acquisition_type",
        }
        assert required.issubset(set(df.columns))


# ── get_current_matchup tests ─────────────────────────────────────────────────


def _matchup_json() -> dict:  # type: ignore[type-arg]
    return {
        "fantasy_content": {
            "team": [
                [{"team_key": "423.l.87941.t.1"}],
                {
                    "matchups": {
                        "0": {
                            "matchup": {
                                "week": "1",
                                "week_start": "2026-04-07",
                                "status": "postevent",
                                "teams": {
                                    "0": {
                                        "team": [
                                            [{"team_key": "423.l.87941.t.1"}],
                                            {
                                                "team_stats": {
                                                    "stats": [
                                                        {
                                                            "stat": {
                                                                "stat_id": "7",
                                                                "value": "42",
                                                            }
                                                        },
                                                        {
                                                            "stat": {
                                                                "stat_id": "12",
                                                                "value": "8",
                                                            }
                                                        },
                                                        {
                                                            "stat": {
                                                                "stat_id": "28",
                                                                "value": "3",
                                                            }
                                                        },
                                                        {
                                                            "stat": {
                                                                "stat_id": "42",
                                                                "value": "55",
                                                            }
                                                        },
                                                        {
                                                            "stat": {
                                                                "stat_id": "48",
                                                                "value": "1.18",
                                                            }
                                                        },
                                                    ]
                                                }
                                            },
                                        ]
                                    },
                                    "1": {
                                        "team": [
                                            [{"team_key": "423.l.87941.t.2"}],
                                            {
                                                "team_stats": {
                                                    "stats": [
                                                        {
                                                            "stat": {
                                                                "stat_id": "7",
                                                                "value": "38",
                                                            }
                                                        },
                                                        {
                                                            "stat": {
                                                                "stat_id": "12",
                                                                "value": "9",
                                                            }
                                                        },
                                                        {
                                                            "stat": {
                                                                "stat_id": "28",
                                                                "value": "2",
                                                            }
                                                        },
                                                        {
                                                            "stat": {
                                                                "stat_id": "42",
                                                                "value": "48",
                                                            }
                                                        },
                                                        {
                                                            "stat": {
                                                                "stat_id": "48",
                                                                "value": "1.32",
                                                            }
                                                        },
                                                    ]
                                                }
                                            },
                                        ]
                                    },
                                    "count": 2,
                                },
                            }
                        },
                        "count": 1,
                    }
                },
            ]
        }
    }


class TestGetCurrentMatchup:
    @responses_lib.activate
    def test_returns_dataframe_with_correct_columns(
        self, client: YahooClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("YAHOO_LEAGUE_ID", "87941")
        monkeypatch.setenv("YAHOO_TEAM_ID", "1")

        responses_lib.add(
            responses_lib.GET,
            BASE + "team/423.l.87941.t.1/matchups",
            status=200,
            json=_matchup_json(),
        )

        df = client.get_current_matchup()
        assert isinstance(df, pd.DataFrame)
        required = {
            "matchup_id",
            "league_id",
            "week_number",
            "season",
            "team_id_home",
            "team_id_away",
        }
        assert required.issubset(set(df.columns))

    @responses_lib.activate
    def test_stat_columns_present(
        self, client: YahooClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("YAHOO_LEAGUE_ID", "87941")
        monkeypatch.setenv("YAHOO_TEAM_ID", "1")

        responses_lib.add(
            responses_lib.GET,
            BASE + "team/423.l.87941.t.1/matchups",
            status=200,
            json=_matchup_json(),
        )

        df = client.get_current_matchup()
        for col in ("h_home", "h_away", "hr_home", "whip_home", "whip_away"):
            assert col in df.columns, f"Missing column: {col}"


# ── get_free_agents tests ─────────────────────────────────────────────────────


def _free_agents_json() -> dict:  # type: ignore[type-arg]
    return {
        "fantasy_content": {
            "league": [
                [{"league_id": "87941"}],
                {
                    "players": {
                        "0": {
                            "player": [
                                [
                                    {"player_key": "423.p.9999"},
                                    {"name": {"full": "Test Player"}},
                                    {"editorial_team_abbr": "NYY"},
                                    {"status": "Active"},
                                    {"eligible_positions": [{"position": "OF"}]},
                                    {"player_id": "123456"},
                                ],
                                {
                                    "player_stats": {
                                        "stats": [
                                            {"stat": {"stat_id": "7", "value": "30"}},
                                            {"stat": {"stat_id": "12", "value": "5"}},
                                        ]
                                    }
                                },
                            ]
                        },
                        "count": 1,
                    }
                },
            ]
        }
    }


class TestGetFreeAgents:
    @responses_lib.activate
    def test_returns_dataframe_with_correct_columns(
        self, client: YahooClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("YAHOO_LEAGUE_ID", "87941")

        responses_lib.add(
            responses_lib.GET,
            BASE + "league/423.l.87941/players;status=FA;count=1",
            status=200,
            json=_free_agents_json(),
        )

        df = client.get_free_agents(count=1)
        assert isinstance(df, pd.DataFrame)
        required = {"player_id", "full_name", "team", "positions", "status"}
        assert required.issubset(set(df.columns))

    @responses_lib.activate
    def test_player_data_parsed_correctly(
        self, client: YahooClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("YAHOO_LEAGUE_ID", "87941")

        responses_lib.add(
            responses_lib.GET,
            BASE + "league/423.l.87941/players;status=FA;count=1",
            status=200,
            json=_free_agents_json(),
        )

        df = client.get_free_agents(count=1)
        assert len(df) == 1
        assert df.iloc[0]["player_id"] == "423.p.9999"
        assert df.iloc[0]["team"] == "NYY"


# ── get_transactions tests ────────────────────────────────────────────────────


def _transactions_json() -> dict:  # type: ignore[type-arg]
    import time as t

    recent_ts = int(t.time()) - 3600  # 1 hour ago (within 7-day window)
    return {
        "fantasy_content": {
            "league": [
                [{"league_id": "87941"}],
                {
                    "transactions": {
                        "0": {
                            "transaction": [
                                [
                                    {"transaction_key": "423.l.87941.tr.1"},
                                    {"type": "add"},
                                    {"timestamp": str(recent_ts)},
                                    {
                                        "players": {
                                            "0": {
                                                "player": [
                                                    [{"player_key": "423.p.7578"}],
                                                    {
                                                        "transaction_data": [
                                                            {
                                                                "transaction_data": {
                                                                    "type": "add",
                                                                    "destination_team_key": "423.l.87941.t.3",
                                                                    "source_team_key": None,
                                                                }
                                                            }
                                                        ]
                                                    },
                                                ]
                                            },
                                            "count": 1,
                                        }
                                    },
                                ]
                            ]
                        },
                        "count": 1,
                    }
                },
            ]
        }
    }


class TestGetTransactions:
    @responses_lib.activate
    def test_returns_dataframe_with_correct_columns(
        self, client: YahooClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("YAHOO_LEAGUE_ID", "87941")

        responses_lib.add(
            responses_lib.GET,
            BASE + "league/423.l.87941/transactions;types=add,drop,trade;count=50",
            status=200,
            json=_transactions_json(),
        )

        df = client.get_transactions(days=7)
        assert isinstance(df, pd.DataFrame)
        required = {
            "transaction_id",
            "league_id",
            "transaction_date",
            "type",
            "team_id",
            "player_id",
            "from_team_id",
            "notes",
        }
        assert required.issubset(set(df.columns))


# ── get_player_details tests ──────────────────────────────────────────────────


def _player_details_json() -> dict:  # type: ignore[type-arg]
    return {
        "fantasy_content": {
            "players": {
                "0": {
                    "player": [
                        [
                            {"player_key": "423.p.7578"},
                            {"player_id": "660670"},
                            {"name": {"full": "Yordan Alvarez"}},
                            {"editorial_team_abbr": "HOU"},
                            {"status": "Active"},
                            {
                                "eligible_positions": [
                                    {"position": "OF"},
                                    {"position": "1B"},
                                ]
                            },
                            {"batting_hand": "L"},
                            {"throwing_hand": "R"},
                        ]
                    ]
                },
                "count": 1,
            }
        }
    }


class TestGetPlayerDetails:
    @responses_lib.activate
    def test_returns_dataframe_with_correct_columns(self, client: YahooClient) -> None:
        responses_lib.add(
            responses_lib.GET,
            BASE + "players;player_keys=423.p.7578",
            status=200,
            json=_player_details_json(),
        )

        df = client.get_player_details(["423.p.7578"])
        assert isinstance(df, pd.DataFrame)
        required = {
            "player_id",
            "mlb_id",
            "full_name",
            "team",
            "positions",
            "bats",
            "throws",
            "status",
            "updated_at",
        }
        assert required.issubset(set(df.columns))

    @responses_lib.activate
    def test_player_data_parsed_correctly(self, client: YahooClient) -> None:
        responses_lib.add(
            responses_lib.GET,
            BASE + "players;player_keys=423.p.7578",
            status=200,
            json=_player_details_json(),
        )

        df = client.get_player_details(["423.p.7578"])
        assert len(df) == 1
        row = df.iloc[0]
        assert row["player_id"] == "423.p.7578"
        assert row["full_name"] == "Yordan Alvarez"
        assert row["team"] == "HOU"
        assert row["bats"] == "L"

    def test_empty_list_returns_empty_dataframe(self, client: YahooClient) -> None:
        df = client.get_player_details([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "player_id" in df.columns


# ── get_standings tests ───────────────────────────────────────────────────────


def _standings_json() -> dict:  # type: ignore[type-arg]
    return {
        "fantasy_content": {
            "league": [
                [{"league_id": "87941"}],
                {
                    "standings": [
                        {
                            "teams": {
                                "0": {
                                    "team": [
                                        [
                                            {"team_key": "423.l.87941.t.1"},
                                            {"name": "Murderers' Row"},
                                            {
                                                "team_standings": {
                                                    "rank": 1,
                                                    "outcome_totals": {
                                                        "wins": "50",
                                                        "losses": "20",
                                                        "ties": "0",
                                                    },
                                                }
                                            },
                                        ]
                                    ]
                                },
                                "count": 1,
                            }
                        }
                    ]
                },
            ]
        }
    }


class TestGetStandings:
    @responses_lib.activate
    def test_returns_dataframe_with_correct_columns(
        self, client: YahooClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("YAHOO_LEAGUE_ID", "87941")

        responses_lib.add(
            responses_lib.GET,
            BASE + "league/423.l.87941/standings",
            status=200,
            json=_standings_json(),
        )

        df = client.get_standings()
        assert isinstance(df, pd.DataFrame)
        required = {"team_id", "team_name", "wins", "losses", "ties", "rank"}
        assert required.issubset(set(df.columns))

    @responses_lib.activate
    def test_standings_data_parsed_correctly(
        self, client: YahooClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("YAHOO_LEAGUE_ID", "87941")

        responses_lib.add(
            responses_lib.GET,
            BASE + "league/423.l.87941/standings",
            status=200,
            json=_standings_json(),
        )

        df = client.get_standings()
        assert len(df) == 1
        row = df.iloc[0]
        assert row["team_id"] == "423.l.87941.t.1"
        assert row["team_name"] == "Murderers' Row"
        assert row["wins"] == 50
        assert row["rank"] == 1


# ── Parser edge-case tests ────────────────────────────────────────────────────


class TestParsers:
    def test_parse_roster_response_empty_data(self) -> None:
        df = _parse_roster_response({}, week=1)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_parse_matchup_response_empty_data(self) -> None:
        df = _parse_matchup_response({}, league_key="423.l.87941")
        assert isinstance(df, pd.DataFrame)
        assert "matchup_id" in df.columns

    def test_parse_free_agents_response_empty_data(self) -> None:
        df = _parse_free_agents_response({})
        assert isinstance(df, pd.DataFrame)
        assert "player_id" in df.columns

    def test_parse_transactions_response_filters_old(self) -> None:
        # Timestamp from 30 days ago — should be filtered out with days=7
        import time as t

        old_ts = int(t.time()) - 30 * 86400
        data = {
            "fantasy_content": {
                "league": [
                    [{"league_id": "87941"}],
                    {
                        "transactions": {
                            "0": {
                                "transaction": [
                                    [
                                        {"transaction_key": "423.l.87941.tr.99"},
                                        {"type": "add"},
                                        {"timestamp": str(old_ts)},
                                        {
                                            "players": {
                                                "0": {
                                                    "player": [
                                                        [{"player_key": "423.p.0001"}],
                                                        {"transaction_data": []},
                                                    ]
                                                },
                                                "count": 1,
                                            }
                                        },
                                    ]
                                ]
                            },
                            "count": 1,
                        }
                    },
                ]
            }
        }
        df = _parse_transactions_response(data, days=7)
        assert len(df) == 0

    def test_parse_player_details_returns_list(self) -> None:
        rows = _parse_player_details(_player_details_json())
        assert isinstance(rows, list)
        assert len(rows) == 1
        assert rows[0]["player_id"] == "423.p.7578"

    def test_parse_standings_response_empty_data(self) -> None:
        df = _parse_standings_response({})
        assert isinstance(df, pd.DataFrame)
        assert "team_id" in df.columns
