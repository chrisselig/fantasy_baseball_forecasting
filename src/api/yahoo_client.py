"""
yahoo_client.py

Authenticated Yahoo Fantasy Sports API client.

All HTTP calls are made against the Yahoo Fantasy Sports API v2.
JSON format is requested via the ``format=json`` query param.

Base URL: https://fantasysports.yahooapis.com/fantasy/v2/

Authentication uses OAuth 2.0.  The initial token pair (access_token +
refresh_token) is generated once locally via scripts/yahoo_auth.py and
stored in GitHub Secrets.  The pipeline refreshes the access token
automatically on every run because Yahoo tokens expire after one hour.

Environment variables consumed by ``YahooClient.from_env()``:
    YAHOO_CONSUMER_KEY
    YAHOO_CONSUMER_SECRET
    YAHOO_ACCESS_TOKEN
    YAHOO_REFRESH_TOKEN
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_URL = "https://fantasysports.yahooapis.com/fantasy/v2/"
TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"

# Yahoo access tokens expire after 3600 seconds.  Refresh if fewer than
# TOKEN_EXPIRY_BUFFER_SECONDS remain on the clock.
_TOKEN_LIFETIME_SECONDS = 3600
_TOKEN_EXPIRY_BUFFER_SECONDS = 300  # 5-minute safety margin


# ── Custom exceptions ─────────────────────────────────────────────────────────


class YahooAuthError(Exception):
    """Raised when Yahoo returns 401 or 403 (authentication / authorisation)."""


class YahooAPIError(Exception):
    """Raised when Yahoo returns any other non-2xx HTTP status code."""


# ── Client ────────────────────────────────────────────────────────────────────


class YahooClient:
    """Authenticated Yahoo Fantasy Sports API client.

    Construct via :meth:`from_env` in production pipelines, or directly
    in tests by passing token strings.

    Args:
        consumer_key: Yahoo application consumer key.
        consumer_secret: Yahoo application consumer secret.
        access_token: OAuth 2.0 access token (expires after 1 hour).
        refresh_token: OAuth 2.0 refresh token (long-lived).
    """

    def __init__(
        self,
        consumer_key: str,
        consumer_secret: str,
        access_token: str,
        refresh_token: str,
    ) -> None:
        self._consumer_key = consumer_key
        self._consumer_secret = consumer_secret
        self._access_token = access_token
        self._refresh_token = refresh_token
        # Track when the current access token was issued so we can pre-emptively
        # refresh before it actually expires.
        self._token_issued_at: float = datetime.now(tz=UTC).timestamp()

    # ── Construction helpers ──────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> YahooClient:
        """Construct from environment variables.

        Reads:
            YAHOO_CONSUMER_KEY
            YAHOO_CONSUMER_SECRET
            YAHOO_ACCESS_TOKEN
            YAHOO_REFRESH_TOKEN

        Returns:
            Fully initialised :class:`YahooClient`.

        Raises:
            EnvironmentError: If any of the four required variables are absent.
        """
        required = {
            "YAHOO_CONSUMER_KEY",
            "YAHOO_CONSUMER_SECRET",
            "YAHOO_ACCESS_TOKEN",
            "YAHOO_REFRESH_TOKEN",
        }
        missing = [k for k in required if not os.environ.get(k)]
        if missing:
            raise OSError(
                f"Missing required environment variables: {', '.join(sorted(missing))}"
            )

        return cls(
            consumer_key=os.environ["YAHOO_CONSUMER_KEY"],
            consumer_secret=os.environ["YAHOO_CONSUMER_SECRET"],
            access_token=os.environ["YAHOO_ACCESS_TOKEN"],
            refresh_token=os.environ["YAHOO_REFRESH_TOKEN"],
        )

    # ── Token management ──────────────────────────────────────────────────────

    def _token_age_seconds(self) -> float:
        """Return how many seconds the current access token has been alive."""
        return datetime.now(tz=UTC).timestamp() - self._token_issued_at

    def _refresh_token_if_needed(self) -> None:
        """Refresh the access token if it is about to expire.

        Yahoo access tokens are valid for 3600 seconds.  We pre-emptively
        refresh when fewer than ``_TOKEN_EXPIRY_BUFFER_SECONDS`` remain so
        that in-flight requests never hit an expired token.
        """
        age = self._token_age_seconds()
        if age < (_TOKEN_LIFETIME_SECONDS - _TOKEN_EXPIRY_BUFFER_SECONDS):
            return  # token is still fresh

        logger.info("Access token is %d seconds old — refreshing.", int(age))
        response = requests.post(
            TOKEN_URL,
            auth=(self._consumer_key, self._consumer_secret),
            data={
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token,
            },
            timeout=30,
        )

        if response.status_code == 401:
            raise YahooAuthError(f"Token refresh failed (401): {response.text}")
        if not response.ok:
            raise YahooAPIError(
                f"Token refresh failed ({response.status_code}): {response.text}"
            )

        data = response.json()
        self._access_token = data["access_token"]
        # The refresh token may rotate; keep the latest one.
        self._refresh_token = data.get("refresh_token", self._refresh_token)
        self._token_issued_at = datetime.now(tz=UTC).timestamp()
        logger.info("Access token refreshed successfully.")

    # ── Base request ──────────────────────────────────────────────────────────

    def _get(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Authenticated GET request to the Yahoo Fantasy Sports API.

        Args:
            endpoint: Path relative to ``BASE_URL``, e.g.
                      ``"league/nfl.l.12345/scoreboard"``.
            params: Additional query parameters.  ``format=json`` is added
                    automatically.

        Returns:
            Parsed JSON response body as a dict.

        Raises:
            YahooAuthError: On HTTP 401 or 403.
            YahooAPIError: On any other non-2xx HTTP status.
        """
        self._refresh_token_if_needed()

        url = BASE_URL + endpoint
        merged_params: dict[str, Any] = {"format": "json"}
        if params:
            merged_params.update(params)

        logger.debug("GET %s params=%s", url, merged_params)

        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {self._access_token}"},
            params=merged_params,
            timeout=30,
        )

        logger.info("Response status: %d for %s", response.status_code, url)

        if response.status_code in (401, 403):
            raise YahooAuthError(
                f"Auth error {response.status_code} for {url}: {response.text}"
            )
        if not response.ok:
            raise YahooAPIError(
                f"API error {response.status_code} for {url}: {response.text}"
            )

        return response.json()  # type: ignore[no-any-return]

    # ── Data fetch methods ────────────────────────────────────────────────────

    def _league_key(self) -> str:
        """Return the Yahoo league key (game_key.l.league_id).

        The game key for MLB fantasy in Yahoo is '422'.
        The league ID is read from the YAHOO_LEAGUE_ID env var if set,
        otherwise falls back to the hard-coded Vlad Guerrero Invitational ID.
        Never hardcodes the ID directly in any API call.
        """
        league_id = os.environ.get("YAHOO_LEAGUE_ID", "87941")
        return f"422.l.{league_id}"

    def _my_team_key(self) -> str:
        """Return the Yahoo team key for the authenticated user's team.

        Read from YAHOO_TEAM_ID env var (just the integer part).
        """
        team_id = os.environ.get("YAHOO_TEAM_ID", "1")
        return f"{self._league_key()}.t.{team_id}"

    # ── Roster ────────────────────────────────────────────────────────────────

    def get_my_roster(self, week: int) -> pd.DataFrame:
        """Fetch the authenticated user's roster for a given fantasy week.

        Args:
            week: Fantasy week number (1–25).

        Returns:
            DataFrame with columns matching ``fact_rosters``:
            team_id, player_id, snapshot_date, roster_slot, acquisition_type.
        """
        team_key = self._my_team_key()
        data = self._get(
            f"team/{team_key}/roster",
            params={"week": week},
        )
        return _parse_roster_response(data, week)

    def get_all_rosters(self, week: int) -> pd.DataFrame:
        """Fetch all teams' rosters for a given fantasy week.

        Args:
            week: Fantasy week number (1–25).

        Returns:
            DataFrame with columns matching ``fact_rosters``:
            team_id, player_id, snapshot_date, roster_slot, acquisition_type.
            Concatenated across all teams.
        """
        league_key = self._league_key()
        data = self._get(
            f"league/{league_key}/teams/roster",
            params={"week": week},
        )
        return _parse_all_rosters_response(data, week)

    # ── Matchup ───────────────────────────────────────────────────────────────

    def get_current_matchup(self) -> pd.DataFrame:
        """Fetch the current week's matchup stats for the authenticated team.

        Returns:
            DataFrame with columns matching ``fact_matchups``:
            matchup_id, league_id, week_number, season, team_id_home,
            team_id_away, plus all stat columns (h_home, hr_home, …).
        """
        league_key = self._league_key()
        team_key = self._my_team_key()
        data = self._get(f"team/{team_key}/matchups")
        return _parse_matchup_response(data, league_key)

    # ── Free agents ───────────────────────────────────────────────────────────

    def get_free_agents(self, count: int = 100) -> pd.DataFrame:
        """Fetch available free agents with their current stats.

        Args:
            count: Maximum number of free agents to return (default 100).

        Returns:
            DataFrame for staging (waiver scoring input):
            player_id, full_name, team, positions, status,
            plus available stat columns.
        """
        league_key = self._league_key()
        data = self._get(
            f"league/{league_key}/players",
            params={
                "status": "FA",
                "count": count,
                "out": "stats",
            },
        )
        return _parse_free_agents_response(data)

    # ── Transactions ──────────────────────────────────────────────────────────

    def get_transactions(self, days: int = 7) -> pd.DataFrame:
        """Fetch recent league transactions (adds, drops, trades).

        Args:
            days: Look-back window in days (default 7).

        Returns:
            DataFrame with columns matching ``fact_transactions``:
            transaction_id, league_id, transaction_date, type, team_id,
            player_id, from_team_id, notes.
        """
        league_key = self._league_key()
        data = self._get(
            f"league/{league_key}/transactions",
            params={"types": "add,drop,trade", "count": 50},
        )
        return _parse_transactions_response(data, days)

    # ── Player details ────────────────────────────────────────────────────────

    def get_player_details(self, player_ids: list[str]) -> pd.DataFrame:
        """Fetch detailed player information for a list of Yahoo player keys.

        Args:
            player_ids: List of Yahoo player keys, e.g. ``['422.p.7578']``.

        Returns:
            DataFrame with columns matching ``dim_players``:
            player_id, mlb_id, full_name, team, positions, bats, throws,
            status, updated_at.
        """
        # Yahoo supports up to 25 player keys per request; batch if needed.
        all_rows: list[dict[str, Any]] = []
        batch_size = 25
        for i in range(0, max(len(player_ids), 1), batch_size):
            batch = player_ids[i : i + batch_size]
            if not batch:
                break
            keys_param = ",".join(batch)
            data = self._get(
                f"players;player_keys={keys_param}",
                params={"out": "metadata"},
            )
            all_rows.extend(_parse_player_details(data))

        if not all_rows:
            return pd.DataFrame(
                columns=[
                    "player_id",
                    "mlb_id",
                    "full_name",
                    "team",
                    "positions",
                    "bats",
                    "throws",
                    "status",
                    "updated_at",
                ]
            )
        return pd.DataFrame(all_rows)

    # ── Standings ─────────────────────────────────────────────────────────────

    def get_standings(self) -> pd.DataFrame:
        """Fetch current league standings.

        Returns:
            DataFrame with columns:
            team_id, team_name, wins, losses, ties, rank.
        """
        league_key = self._league_key()
        data = self._get(f"league/{league_key}/standings")
        return _parse_standings_response(data)


# ── Response parsers ──────────────────────────────────────────────────────────
# These functions normalise raw Yahoo JSON into DataFrames whose columns
# exactly match the names in src/db/schema.py.
#
# Yahoo's JSON structure is deeply nested.  The parsers navigate this
# structure defensively — missing keys fall back to None.


def _safe_get(obj: Any, *keys: str | int, default: Any = None) -> Any:
    """Safely navigate a nested dict/list, returning ``default`` if any key is absent."""
    for key in keys:
        if isinstance(obj, dict):
            obj = obj.get(key, default)
        elif isinstance(obj, (list, tuple)) and isinstance(key, int):
            try:
                obj = obj[key]
            except IndexError:
                return default
        else:
            return default
        if obj is default:
            return default
    return obj


def _parse_roster_response(data: dict[str, Any], week: int) -> pd.DataFrame:
    """Parse a single-team roster response into the fact_rosters shape."""
    rows: list[dict[str, Any]] = []
    snapshot = datetime.now(tz=UTC).date().isoformat()

    try:
        team_data = data["fantasy_content"]["team"]
        # team_data[0] is a list with team metadata dicts
        team_meta = team_data[0]
        team_key = ""
        for item in team_meta:
            if isinstance(item, dict) and "team_key" in item:
                team_key = item["team_key"]
                break

        roster_players = _safe_get(team_data, 1, "roster", "0", "players", default={})
        if not isinstance(roster_players, dict):
            roster_players = {}

        for _k, player_entry in roster_players.items():
            if _k == "count":
                continue
            if not isinstance(player_entry, dict):
                continue
            player = player_entry.get("player", [])
            if not player:
                continue

            player_meta: list[Any] = player[0] if isinstance(player, list) else []
            selected_position = _safe_get(player, 1, "selected_position", default=[])

            player_key = ""
            acquisition_type = None
            roster_slot = "BN"

            for item in player_meta:
                if isinstance(item, dict):
                    if "player_key" in item:
                        player_key = item["player_key"]
                    if "acquisition_type" in item:
                        acquisition_type = item["acquisition_type"]

            for sp_item in selected_position:
                if isinstance(sp_item, dict) and "position" in sp_item:
                    roster_slot = sp_item["position"]

            if player_key:
                rows.append(
                    {
                        "team_id": team_key,
                        "player_id": player_key,
                        "snapshot_date": snapshot,
                        "roster_slot": roster_slot,
                        "acquisition_type": acquisition_type,
                    }
                )
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("Failed to parse roster response: %s", exc)

    return pd.DataFrame(
        rows,
        columns=[
            "team_id",
            "player_id",
            "snapshot_date",
            "roster_slot",
            "acquisition_type",
        ],
    )


def _parse_all_rosters_response(data: dict[str, Any], week: int) -> pd.DataFrame:
    """Parse a league-wide roster response (all teams) into the fact_rosters shape."""
    frames: list[pd.DataFrame] = []

    try:
        teams = _safe_get(data, "fantasy_content", "league", 1, "teams", default={})
        if not isinstance(teams, dict):
            return pd.DataFrame(
                columns=[
                    "team_id",
                    "player_id",
                    "snapshot_date",
                    "roster_slot",
                    "acquisition_type",
                ]
            )

        for _k, team_entry in teams.items():
            if _k == "count":
                continue
            if not isinstance(team_entry, dict):
                continue
            team = team_entry.get("team", [])
            # Re-use single-team parser by wrapping in the expected structure
            wrapped = {"fantasy_content": {"team": team}}
            frame = _parse_roster_response(wrapped, week)
            frames.append(frame)
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("Failed to parse all-rosters response: %s", exc)

    if not frames:
        return pd.DataFrame(
            columns=[
                "team_id",
                "player_id",
                "snapshot_date",
                "roster_slot",
                "acquisition_type",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _parse_matchup_response(data: dict[str, Any], league_key: str) -> pd.DataFrame:
    """Parse matchup response into the fact_matchups shape."""
    rows: list[dict[str, Any]] = []

    try:
        matchups = _safe_get(data, "fantasy_content", "team", 1, "matchups", default={})
        if not isinstance(matchups, dict):
            return _empty_matchup_df()

        league_id = int(league_key.split(".")[-1])

        for _k, matchup_entry in matchups.items():
            if _k == "count":
                continue
            if not isinstance(matchup_entry, dict):
                continue
            matchup = matchup_entry.get("matchup", {})
            if not matchup:
                continue

            week_number = int(matchup.get("week", 0))
            season = int(matchup.get("week_start", "2026-01-01")[:4])

            teams_data = matchup.get("teams", {})
            team_keys: list[str] = []
            team_stats: list[dict[str, Any]] = []

            for tk, team_entry in teams_data.items():
                if tk == "count":
                    continue
                if not isinstance(team_entry, dict):
                    continue
                team = team_entry.get("team", [])
                if not team:
                    continue
                team_meta: list[Any] = team[0] if isinstance(team, list) else []
                tkey = ""
                for item in team_meta:
                    if isinstance(item, dict) and "team_key" in item:
                        tkey = item["team_key"]
                        break
                team_keys.append(tkey)

                stats_raw = _safe_get(team, 1, "team_stats", "stats", default=[])
                stats_dict: dict[str, float] = {}
                for stat_entry in stats_raw:
                    if isinstance(stat_entry, dict):
                        stat = stat_entry.get("stat", {})
                        sid = stat.get("stat_id", "")
                        val = stat.get("value", None)
                        if sid and val is not None:
                            try:
                                stats_dict[str(sid)] = float(val)
                            except (ValueError, TypeError):
                                stats_dict[str(sid)] = 0.0
                team_stats.append(stats_dict)

            if len(team_keys) < 2:
                continue

            home_key, away_key = team_keys[0], team_keys[1]
            home_stats = team_stats[0] if team_stats else {}
            away_stats = team_stats[1] if len(team_stats) > 1 else {}

            matchup_id = (
                f"{league_id}_{season}_W{week_number:02d}_{home_key}vs{away_key}"
            )

            row: dict[str, Any] = {
                "matchup_id": matchup_id,
                "league_id": league_id,
                "week_number": week_number,
                "season": season,
                "team_id_home": home_key,
                "team_id_away": away_key,
                # Stat columns — Yahoo stat IDs mapped to schema column names
                # Batter stats
                "h_home": home_stats.get("7"),
                "h_away": away_stats.get("7"),
                "hr_home": home_stats.get("12"),
                "hr_away": away_stats.get("12"),
                "sb_home": home_stats.get("16"),
                "sb_away": away_stats.get("16"),
                "bb_home": home_stats.get("13"),
                "bb_away": away_stats.get("13"),
                "avg_home": home_stats.get("3"),
                "avg_away": away_stats.get("3"),
                "ops_home": home_stats.get("55"),
                "ops_away": away_stats.get("55"),
                "fpct_home": home_stats.get("23"),
                "fpct_away": away_stats.get("23"),
                # Pitcher stats
                "w_home": home_stats.get("28"),
                "w_away": away_stats.get("28"),
                "k_home": home_stats.get("42"),
                "k_away": away_stats.get("42"),
                "whip_home": home_stats.get("48"),
                "whip_away": away_stats.get("48"),
                "k_bb_home": home_stats.get("26"),
                "k_bb_away": away_stats.get("26"),
                "sv_h_home": home_stats.get("57"),
                "sv_h_away": away_stats.get("57"),
                "categories_won_home": None,
                "categories_won_away": None,
                "result": None,
            }
            rows.append(row)
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("Failed to parse matchup response: %s", exc)

    if not rows:
        return _empty_matchup_df()
    return pd.DataFrame(rows)


def _empty_matchup_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "matchup_id",
            "league_id",
            "week_number",
            "season",
            "team_id_home",
            "team_id_away",
            "h_home",
            "h_away",
            "hr_home",
            "hr_away",
            "sb_home",
            "sb_away",
            "bb_home",
            "bb_away",
            "avg_home",
            "avg_away",
            "ops_home",
            "ops_away",
            "fpct_home",
            "fpct_away",
            "w_home",
            "w_away",
            "k_home",
            "k_away",
            "whip_home",
            "whip_away",
            "k_bb_home",
            "k_bb_away",
            "sv_h_home",
            "sv_h_away",
            "categories_won_home",
            "categories_won_away",
            "result",
        ]
    )


def _parse_free_agents_response(data: dict[str, Any]) -> pd.DataFrame:
    """Parse free agent list response into a staging DataFrame."""
    rows: list[dict[str, Any]] = []

    try:
        players = _safe_get(data, "fantasy_content", "league", 1, "players", default={})
        if not isinstance(players, dict):
            return _empty_free_agents_df()

        for _k, player_entry in players.items():
            if _k == "count":
                continue
            if not isinstance(player_entry, dict):
                continue
            player = player_entry.get("player", [])
            if not player:
                continue

            meta: list[Any] = player[0] if isinstance(player, list) else []
            player_key = ""
            full_name = ""
            team = ""
            positions: list[str] = []
            status = "Active"
            mlb_id = None

            for item in meta:
                if not isinstance(item, dict):
                    continue
                if "player_key" in item:
                    player_key = item["player_key"]
                if "full_name" in item:
                    full_name = item["full_name"]
                if "editorial_team_abbr" in item:
                    team = item["editorial_team_abbr"]
                if "status" in item:
                    status = item["status"]
                if "eligible_positions" in item:
                    for ep in item["eligible_positions"]:
                        if isinstance(ep, dict) and "position" in ep:
                            positions.append(ep["position"])
                if "player_id" in item:
                    try:
                        mlb_id = int(item["player_id"])
                    except (ValueError, TypeError):
                        mlb_id = None

            # Stats (from "out=stats" param)
            stats_raw = _safe_get(player, 1, "player_stats", "stats", default=[])
            stats_dict: dict[str, float] = {}
            for stat_entry in stats_raw:
                if isinstance(stat_entry, dict):
                    stat = stat_entry.get("stat", {})
                    sid = str(stat.get("stat_id", ""))
                    val = stat.get("value", None)
                    if sid and val is not None:
                        try:
                            stats_dict[sid] = float(val)
                        except (ValueError, TypeError):
                            stats_dict[sid] = 0.0

            if player_key:
                rows.append(
                    {
                        "player_id": player_key,
                        "full_name": full_name,
                        "team": team,
                        "positions": positions,
                        "status": status,
                        "mlb_id": mlb_id,
                        **{f"stat_{k}": v for k, v in stats_dict.items()},
                    }
                )
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("Failed to parse free agents response: %s", exc)

    if not rows:
        return _empty_free_agents_df()
    return pd.DataFrame(rows)


def _empty_free_agents_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["player_id", "full_name", "team", "positions", "status"]
    )


def _parse_transactions_response(data: dict[str, Any], days: int) -> pd.DataFrame:
    """Parse transactions response into the fact_transactions shape."""
    rows: list[dict[str, Any]] = []
    cutoff_ts = datetime.now(tz=UTC).timestamp() - days * 86400

    try:
        league_data = _safe_get(data, "fantasy_content", "league", default=[])
        league_id = 0
        if isinstance(league_data, list) and league_data:
            for item in league_data[0]:
                if isinstance(item, dict) and "league_id" in item:
                    league_id = int(item["league_id"])
                    break

        transactions = _safe_get(
            data, "fantasy_content", "league", 1, "transactions", default={}
        )
        if not isinstance(transactions, dict):
            return _empty_transactions_df()

        for _k, txn_entry in transactions.items():
            if _k == "count":
                continue
            if not isinstance(txn_entry, dict):
                continue
            txn = txn_entry.get("transaction", [])
            if not txn:
                continue

            meta: list[Any] = txn[0] if isinstance(txn, list) else []
            transaction_id = ""
            txn_type = ""
            timestamp_val = 0.0
            team_id = ""
            from_team_id = None
            player_key = ""
            notes = None

            for item in meta:
                if not isinstance(item, dict):
                    continue
                if "transaction_key" in item:
                    transaction_id = item["transaction_key"]
                if "type" in item:
                    txn_type = item["type"]
                if "timestamp" in item:
                    try:
                        timestamp_val = float(item["timestamp"])
                    except (ValueError, TypeError):
                        timestamp_val = 0.0
                if "players" in item:
                    for _pk, p_entry in item["players"].items():
                        if _pk == "count":
                            continue
                        if not isinstance(p_entry, dict):
                            continue
                        p = p_entry.get("player", [])
                        p_meta = p[0] if isinstance(p, list) and p else []
                        for pm in p_meta:
                            if isinstance(pm, dict) and "player_key" in pm:
                                player_key = pm["player_key"]
                        tx_data = p[1].get("transaction_data", []) if len(p) > 1 else []
                        for td in tx_data:
                            if not isinstance(td, dict):
                                continue
                            txd = td.get("transaction_data", {})
                            if txd.get("type") in ("add", "drop", "trade_accept"):
                                team_id = txd.get("destination_team_key", "")
                                from_team_id = txd.get("source_team_key", None)

            if timestamp_val < cutoff_ts:
                continue

            if transaction_id and player_key:
                txn_dt = datetime.fromtimestamp(timestamp_val, tz=UTC).isoformat()
                rows.append(
                    {
                        "transaction_id": transaction_id,
                        "league_id": league_id,
                        "transaction_date": txn_dt,
                        "type": txn_type,
                        "team_id": team_id,
                        "player_id": player_key,
                        "from_team_id": from_team_id,
                        "notes": notes,
                    }
                )
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("Failed to parse transactions response: %s", exc)

    if not rows:
        return _empty_transactions_df()
    return pd.DataFrame(rows)


def _empty_transactions_df() -> pd.DataFrame:
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


def _parse_player_details(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse a players endpoint response into a list of dim_players rows."""
    rows: list[dict[str, Any]] = []

    try:
        players = _safe_get(data, "fantasy_content", "players", default={})
        if not isinstance(players, dict):
            return rows

        now_str = datetime.now(tz=UTC).isoformat()

        for _k, player_entry in players.items():
            if _k == "count":
                continue
            if not isinstance(player_entry, dict):
                continue
            player = player_entry.get("player", [])
            if not player:
                continue

            meta: list[Any] = player[0] if isinstance(player, list) else []
            player_key = ""
            mlb_id = None
            full_name = ""
            team = ""
            positions: list[str] = []
            bats = None
            throws = None
            status = "Active"

            for item in meta:
                if not isinstance(item, dict):
                    continue
                if "player_key" in item:
                    player_key = item["player_key"]
                if "player_id" in item:
                    try:
                        mlb_id = int(item["player_id"])
                    except (ValueError, TypeError):
                        mlb_id = None
                if "full_name" in item:
                    full_name = item["full_name"]
                if "editorial_team_abbr" in item:
                    team = item["editorial_team_abbr"]
                if "status" in item:
                    status = item["status"]
                if "eligible_positions" in item:
                    for ep in item["eligible_positions"]:
                        if isinstance(ep, dict) and "position" in ep:
                            positions.append(ep["position"])
                if "batting_hand" in item:
                    bats = item["batting_hand"]
                if "throwing_hand" in item:
                    throws = item["throwing_hand"]

            if player_key:
                rows.append(
                    {
                        "player_id": player_key,
                        "mlb_id": mlb_id,
                        "full_name": full_name,
                        "team": team,
                        "positions": positions,
                        "bats": bats,
                        "throws": throws,
                        "status": status,
                        "updated_at": now_str,
                    }
                )
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("Failed to parse player details: %s", exc)

    return rows


def _parse_standings_response(data: dict[str, Any]) -> pd.DataFrame:
    """Parse standings response into a reference DataFrame."""
    rows: list[dict[str, Any]] = []

    try:
        teams = _safe_get(
            data, "fantasy_content", "league", 1, "standings", 0, "teams", default={}
        )
        if not isinstance(teams, dict):
            return pd.DataFrame(
                columns=["team_id", "team_name", "wins", "losses", "ties", "rank"]
            )

        for _k, team_entry in teams.items():
            if _k == "count":
                continue
            if not isinstance(team_entry, dict):
                continue
            team = team_entry.get("team", [])
            if not team:
                continue

            meta: list[Any] = team[0] if isinstance(team, list) else []
            team_key = ""
            team_name = ""
            wins = 0
            losses = 0
            ties = 0
            rank = 0

            for item in meta:
                if not isinstance(item, dict):
                    continue
                if "team_key" in item:
                    team_key = item["team_key"]
                if "name" in item:
                    team_name = item["name"]
                if "team_standings" in item:
                    ts = item["team_standings"]
                    rank = int(ts.get("rank", 0))
                    outcome = ts.get("outcome_totals", {})
                    wins = int(outcome.get("wins", 0))
                    losses = int(outcome.get("losses", 0))
                    ties = int(outcome.get("ties", 0))

            if team_key:
                rows.append(
                    {
                        "team_id": team_key,
                        "team_name": team_name,
                        "wins": wins,
                        "losses": losses,
                        "ties": ties,
                        "rank": rank,
                    }
                )
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("Failed to parse standings response: %s", exc)

    if not rows:
        return pd.DataFrame(
            columns=["team_id", "team_name", "wins", "losses", "ties", "rank"]
        )
    return pd.DataFrame(rows)
