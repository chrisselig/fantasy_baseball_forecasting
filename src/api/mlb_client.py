"""
mlb_client.py

MLB data ingestion client for the fantasy baseball pipeline.

Covers:
  - MLB Stats API (public, no auth): call-ups, player info, game schedule,
    minor league stats, daily boxscore stats, pace-based projections
  - Player ID crosswalk builder (MLBAM IDs)

All external API calls are wrapped to never block the pipeline — failures return
empty DataFrames with correct columns and log a WARNING.

MLB Stats API base: https://statsapi.mlb.com/api/v1/
Note: This is an unofficial but stable endpoint. URLs are pinned as constants.
"""

from __future__ import annotations

import datetime
import logging
import warnings
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── MLB Stats API endpoint constants ─────────────────────────────────────────

_MLB_BASE = "https://statsapi.mlb.com/api/v1"
_TRANSACTIONS_URL = f"{_MLB_BASE}/transactions"
_PEOPLE_URL = f"{_MLB_BASE}/people"
_SCHEDULE_URL = f"{_MLB_BASE}/schedule"

# sportId values for minor league levels
_SPORT_AAA = 11
_SPORT_AA = 12
_SPORT_A_PLUS = 13
_SPORT_A = 14
_MINOR_LEAGUE_LEVELS = [
    (_SPORT_AAA, "AAA"),
    (_SPORT_AA, "AA"),
    (_SPORT_A_PLUS, "A+"),
    (_SPORT_A, "A"),
]

# ── Column definitions (must match schema.py) ─────────────────────────────────

_CALLUP_COLUMNS = ["mlb_id", "full_name", "team", "transaction_date", "from_level"]

_PLAYER_INFO_KEYS = [
    "mlb_id",
    "full_name",
    "team",
    "positions",
    "bats",
    "throws",
    "status",
]

_SCHEDULE_COLUMNS = [
    "mlb_id",
    "game_date",
    "opponent_team",
    "home_away",
    "probable_pitcher",
]

_MINOR_LEAGUE_COLUMNS = [
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
    "whip",  # LOWEST WINS
    "era",
]

# Columns matching fact_player_stats_daily (batter portion).
# player_id is NOT included — callers map mlb_id → player_id via dim_players.
_BATTER_STATS_COLUMNS = [
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

# Columns matching fact_player_stats_daily (pitcher portion)
_PITCHER_STATS_COLUMNS = [
    "mlb_id",
    "stat_date",
    "ip",
    "w",
    "k",
    "walks_allowed",
    "hits_allowed",
    "sv",
    "holds",
    "whip",  # LOWEST WINS
    "k_bb",
    "sv_h",
]

# Columns matching fact_projections
_PROJECTIONS_COLUMNS = [
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
    "proj_whip",  # LOWEST WINS
    "proj_k_bb",
    "games_remaining",
    "source",
]

_CROSSWALK_COLUMNS = ["full_name", "mlb_id", "fg_id"]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _mlb_get(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Perform a GET request against the MLB Stats API.

    Args:
        url: Full endpoint URL.
        params: Optional query parameters.

    Returns:
        Parsed JSON response as a dict.

    Raises:
        requests.HTTPError: On non-2xx responses.
        requests.RequestException: On network errors.
    """
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def _empty_df(columns: list[str]) -> pd.DataFrame:
    """Return an empty DataFrame with the specified columns."""
    return pd.DataFrame(columns=columns)


# ── MLB Stats API functions ───────────────────────────────────────────────────


def get_recent_callups(days: int = 7) -> pd.DataFrame:
    """Fetch recent MLB call-ups from the transactions endpoint.

    Source: GET /transactions?sportId=1&startDate=...&endDate=...
    Filter for typeCode="CU" (call-up) transactions only.

    Args:
        days: Number of days to look back. Defaults to 7.

    Returns:
        DataFrame with columns:
            mlb_id (int), full_name (str), team (str),
            transaction_date (date), from_level (str)
    """
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days)

    params = {
        "sportId": 1,
        "startDate": start_date.strftime("%Y-%m-%d"),
        "endDate": end_date.strftime("%Y-%m-%d"),
    }

    logger.debug("Fetching transactions from %s to %s", start_date, end_date)

    data = _mlb_get(_TRANSACTIONS_URL, params=params)
    transactions = data.get("transactions", [])

    rows: list[dict[str, Any]] = []
    for txn in transactions:
        if txn.get("typeCode") != "CU":
            continue

        person = txn.get("person", {})
        to_team = txn.get("toTeam", {})
        from_org = txn.get("fromOrg", {})

        mlb_id = person.get("id")
        if mlb_id is None:
            continue

        raw_date = txn.get("date", txn.get("effectiveDate", ""))
        try:
            txn_date = datetime.date.fromisoformat(raw_date[:10])
        except (ValueError, IndexError):
            txn_date = None

        rows.append(
            {
                "mlb_id": int(mlb_id),
                "full_name": person.get("fullName", ""),
                "team": to_team.get("abbreviation", to_team.get("name", "")),
                "transaction_date": txn_date,
                "from_level": from_org.get("name", ""),
            }
        )

    df = (
        pd.DataFrame(rows, columns=_CALLUP_COLUMNS)
        if rows
        else _empty_df(_CALLUP_COLUMNS)
    )
    logger.info("Found %d call-up transactions in last %d days.", len(df), days)
    return df


def get_player_info(mlb_id: int) -> dict[str, Any]:
    """Fetch player metadata from MLB Stats API people endpoint.

    Source: GET /people/{mlb_id}

    Args:
        mlb_id: MLBAM player ID.

    Returns:
        Dict with keys:
            mlb_id, full_name, team, positions (list[str]),
            bats, throws, status

    Raises:
        requests.HTTPError: On API error.
        KeyError: If the response is missing expected fields.
    """
    url = f"{_PEOPLE_URL}/{mlb_id}"
    data = _mlb_get(url)

    people = data.get("people", [])
    if not people:
        logger.warning("No player data returned for mlb_id=%d", mlb_id)
        return {
            "mlb_id": mlb_id,
            "full_name": "",
            "team": None,
            "positions": [],
            "bats": None,
            "throws": None,
            "status": None,
        }

    person = people[0]
    current_team = person.get("currentTeam", {})
    primary_position = person.get("primaryPosition", {})
    bat_side = person.get("batSide", {})
    pitch_hand = person.get("pitchHand", {})
    active = person.get("active", False)

    return {
        "mlb_id": int(person.get("id", mlb_id)),
        "full_name": person.get("fullName", ""),
        "team": current_team.get("abbreviation", current_team.get("name")),
        "positions": [primary_position.get("abbreviation", "")],
        "bats": bat_side.get("code"),
        "throws": pitch_hand.get("code"),
        "status": "Active" if active else "Inactive",
    }


def get_daily_game_schedule(date: datetime.date) -> pd.DataFrame:
    """Fetch the MLB game schedule for a given date.

    Source: GET /schedule?sportId=1&date=YYYY-MM-DD

    Args:
        date: The date to fetch the schedule for.

    Returns:
        DataFrame with columns:
            mlb_id (int), game_date (date), opponent_team (str),
            home_away (str), probable_pitcher (str | None)
        Used by the lineup optimizer to know who plays today.
    """
    params = {
        "sportId": 1,
        "date": date.strftime("%Y-%m-%d"),
        "hydrate": "probablePitcher,team",
    }

    logger.debug("Fetching schedule for %s", date)
    data = _mlb_get(_SCHEDULE_URL, params=params)

    rows: list[dict[str, Any]] = []
    for game_date_block in data.get("dates", []):
        for game in game_date_block.get("games", []):
            game_date_val = game.get("officialDate", date.isoformat())
            try:
                game_date_parsed = datetime.date.fromisoformat(game_date_val)
            except ValueError:
                game_date_parsed = date

            home_team = game.get("teams", {}).get("home", {})
            away_team = game.get("teams", {}).get("away", {})

            home_team_info = home_team.get("team", {})
            away_team_info = away_team.get("team", {})

            home_abbrev = home_team_info.get(
                "abbreviation", home_team_info.get("name", "")
            )
            away_abbrev = away_team_info.get(
                "abbreviation", away_team_info.get("name", "")
            )

            home_pitcher = home_team.get("probablePitcher")
            away_pitcher = away_team.get("probablePitcher")

            home_pitcher_name = home_pitcher.get("fullName") if home_pitcher else None
            away_pitcher_name = away_pitcher.get("fullName") if away_pitcher else None

            # Home team row
            home_team_id = home_team_info.get("id")
            if home_team_id:
                rows.append(
                    {
                        "mlb_id": int(home_team_id),
                        "game_date": game_date_parsed,
                        "opponent_team": away_abbrev,
                        "home_away": "home",
                        "probable_pitcher": home_pitcher_name,
                    }
                )

            # Away team row
            away_team_id = away_team_info.get("id")
            if away_team_id:
                rows.append(
                    {
                        "mlb_id": int(away_team_id),
                        "game_date": game_date_parsed,
                        "opponent_team": home_abbrev,
                        "home_away": "away",
                        "probable_pitcher": away_pitcher_name,
                    }
                )

    df = (
        pd.DataFrame(rows, columns=_SCHEDULE_COLUMNS)
        if rows
        else _empty_df(_SCHEDULE_COLUMNS)
    )
    logger.info("Found %d team-game entries for %s.", len(df), date)
    return df


def get_minor_league_stats(mlb_id: int, season: int) -> pd.DataFrame:
    """Fetch minor league stats for a player.

    Source: GET /people/{mlb_id}/stats?stats=season&group=hitting,pitching&sportId=11
    sportId: 11=AAA, 12=AA, 13=A+, 14=A
    Tries each level from AAA down, returns the most recent level with data.

    Args:
        mlb_id: MLBAM player ID.
        season: The MLB season year.

    Returns:
        DataFrame with columns:
            mlb_id, season, level (str), ab, h, hr, sb, bb, avg, ops,
            ip, k, whip (LOWEST WINS), era — whichever are available
        Used to evaluate call-up quality.
    """
    url = f"{_PEOPLE_URL}/{mlb_id}/stats"

    for sport_id, level_name in _MINOR_LEAGUE_LEVELS:
        params: dict[str, Any] = {
            "stats": "season",
            "group": "hitting,pitching",
            "sportId": sport_id,
            "season": season,
        }

        logger.debug(
            "Checking %s stats for mlb_id=%d season=%d (sportId=%d)",
            level_name,
            mlb_id,
            season,
            sport_id,
        )

        try:
            data = _mlb_get(url, params=params)
        except requests.RequestException as exc:
            logger.warning(
                "MLB Stats API error for mlb_id=%d at %s: %s", mlb_id, level_name, exc
            )
            continue

        stats_groups = data.get("stats", [])
        if not stats_groups:
            continue

        rows: list[dict[str, Any]] = []
        for group in stats_groups:
            splits = group.get("splits", [])
            if not splits:
                continue

            for split in splits:
                stat = split.get("stat", {})
                row: dict[str, Any] = {
                    "mlb_id": mlb_id,
                    "season": season,
                    "level": level_name,
                    # Batting
                    "ab": stat.get("atBats"),
                    "h": stat.get("hits"),
                    "hr": stat.get("homeRuns"),
                    "sb": stat.get("stolenBases"),
                    "bb": stat.get("baseOnBalls"),
                    "avg": stat.get("avg"),
                    "ops": stat.get("ops"),
                    # Pitching
                    "ip": stat.get("inningsPitched"),
                    "k": stat.get("strikeOuts"),
                    "whip": stat.get("whip"),  # LOWEST WINS
                    "era": stat.get("era"),
                }
                rows.append(row)

        if rows:
            df = pd.DataFrame(rows, columns=_MINOR_LEAGUE_COLUMNS)
            logger.info(
                "Found %s stats for mlb_id=%d season=%d (%d rows).",
                level_name,
                mlb_id,
                season,
                len(df),
            )
            return df

    logger.info("No minor league stats found for mlb_id=%d season=%d.", mlb_id, season)
    return _empty_df(_MINOR_LEAGUE_COLUMNS)


# ── MLB Stats API: Daily stats & projections ────────────────────────────────

# Boxscore endpoint for individual games
_GAME_URL = "https://statsapi.mlb.com/api/v1/game"


def _fetch_boxscores_for_date(
    date: datetime.date,
) -> list[dict[str, Any]]:
    """Fetch boxscores for all MLB games on a given date.

    Args:
        date: The date to fetch boxscores for.

    Returns:
        List of boxscore dicts from the MLB Stats API.
    """
    params: dict[str, Any] = {
        "sportId": 1,
        "date": date.strftime("%Y-%m-%d"),
    }
    try:
        schedule_data = _mlb_get(_SCHEDULE_URL, params=params)
    except requests.RequestException as exc:
        logger.warning("Schedule fetch failed for %s: %s", date, exc)
        return []

    game_pks: list[int] = []
    for date_block in schedule_data.get("dates", []):
        for game in date_block.get("games", []):
            status = game.get("status", {}).get("abstractGameState", "")
            if status == "Final":
                game_pk = game.get("gamePk")
                if game_pk is not None:
                    game_pks.append(int(game_pk))

    boxscores: list[dict[str, Any]] = []
    for game_pk in game_pks:
        try:
            url = f"{_GAME_URL}/{game_pk}/boxscore"
            boxscore = _mlb_get(url)
            boxscores.append(boxscore)
        except requests.RequestException as exc:
            logger.warning("Boxscore fetch failed for gamePk=%d: %s", game_pk, exc)

    logger.info(
        "Fetched %d boxscores for %s (%d games total).",
        len(boxscores),
        date,
        len(game_pks),
    )
    return boxscores


def _extract_batter_rows(
    boxscores: list[dict[str, Any]],
    stat_date: datetime.date,
) -> list[dict[str, Any]]:
    """Extract batter stat rows from boxscore data.

    Args:
        boxscores: List of boxscore dicts from MLB Stats API.
        stat_date: Date to tag on each row.

    Returns:
        List of dicts with batter stat columns.
    """
    rows: list[dict[str, Any]] = []
    for box in boxscores:
        teams = box.get("teams", {})
        for side in ("home", "away"):
            team_data = teams.get(side, {})
            players = team_data.get("players", {})
            for _player_key, player_data in players.items():
                stats = player_data.get("stats", {})
                batting = stats.get("batting", {})
                fielding = stats.get("fielding", {})
                if not batting:
                    continue

                person = player_data.get("person", {})
                mlb_id = person.get("id")
                if mlb_id is None:
                    continue

                ab = batting.get("atBats", 0)
                if ab == 0 and batting.get("plateAppearances", 0) == 0:
                    continue  # Did not bat

                h = batting.get("hits", 0)
                hr = batting.get("homeRuns", 0)
                doubles = batting.get("doubles", 0)
                triples = batting.get("triples", 0)
                singles = h - doubles - triples - hr
                tb = singles + 2 * doubles + 3 * triples + 4 * hr

                errors_val = fielding.get("errors", 0)
                chances_val = fielding.get("chances", 0)

                rows.append(
                    {
                        "mlb_id": int(mlb_id),
                        "stat_date": stat_date,
                        "ab": ab,
                        "h": h,
                        "hr": hr,
                        "sb": batting.get("stolenBases", 0),
                        "bb": batting.get("baseOnBalls", 0),
                        "hbp": batting.get("hitByPitch", 0),
                        "sf": batting.get("sacFlies", 0),
                        "tb": tb,
                        "avg": None,  # Recomputed from components
                        "ops": None,
                        "fpct": None,
                        "errors": errors_val,
                        "chances": chances_val,
                    }
                )
    return rows


def _extract_pitcher_rows(
    boxscores: list[dict[str, Any]],
    stat_date: datetime.date,
) -> list[dict[str, Any]]:
    """Extract pitcher stat rows from boxscore data.

    Args:
        boxscores: List of boxscore dicts from MLB Stats API.
        stat_date: Date to tag on each row.

    Returns:
        List of dicts with pitcher stat columns.
    """
    rows: list[dict[str, Any]] = []
    for box in boxscores:
        teams = box.get("teams", {})
        for side in ("home", "away"):
            team_data = teams.get(side, {})
            players = team_data.get("players", {})
            for _player_key, player_data in players.items():
                stats = player_data.get("stats", {})
                pitching = stats.get("pitching", {})
                if not pitching:
                    continue

                person = player_data.get("person", {})
                mlb_id = person.get("id")
                if mlb_id is None:
                    continue

                ip_str = pitching.get("inningsPitched", "0")
                try:
                    ip = float(ip_str)
                except (ValueError, TypeError):
                    ip = 0.0

                if ip == 0.0:
                    continue  # Did not pitch

                k = pitching.get("strikeOuts", 0)
                walks_allowed = pitching.get("baseOnBalls", 0)
                hits_allowed = pitching.get("hits", 0)
                sv = pitching.get("saves", 0)
                holds = pitching.get("holds", 0)

                whip = (walks_allowed + hits_allowed) / ip if ip > 0 else None
                k_bb = k / walks_allowed if walks_allowed > 0 else None

                rows.append(
                    {
                        "mlb_id": int(mlb_id),
                        "stat_date": stat_date,
                        "ip": ip,
                        "w": pitching.get("wins", 0),
                        "k": k,
                        "walks_allowed": walks_allowed,
                        "hits_allowed": hits_allowed,
                        "sv": sv,
                        "holds": holds,
                        "whip": whip,
                        "k_bb": k_bb,
                        "sv_h": sv + holds,
                    }
                )
    return rows


def get_batter_stats(
    start_date: datetime.date,
    end_date: datetime.date,
) -> pd.DataFrame:
    """Fetch batter stats from MLB Stats API boxscores.

    Source: MLB Stats API /game/{gamePk}/boxscore for all completed games
    on the given date.

    Args:
        start_date: Start of the date range (used with end_date).
        end_date: End of the date range.

    Returns:
        DataFrame with columns matching fact_player_stats_daily:
            mlb_id, stat_date, ab, h, hr, sb, bb, hbp, sf, tb,
            avg, ops, fpct, errors, chances
    """
    boxscores = _fetch_boxscores_for_date(end_date)
    if not boxscores:
        return _empty_df(_BATTER_STATS_COLUMNS)

    rows = _extract_batter_rows(boxscores, end_date)
    if not rows:
        return _empty_df(_BATTER_STATS_COLUMNS)

    df = pd.DataFrame(rows, columns=_BATTER_STATS_COLUMNS)
    logger.info("Loaded %d batter stat rows for %s.", len(df), end_date)
    return df


def get_pitcher_stats(
    start_date: datetime.date,
    end_date: datetime.date,
) -> pd.DataFrame:
    """Fetch pitcher stats from MLB Stats API boxscores.

    Source: MLB Stats API /game/{gamePk}/boxscore for all completed games
    on the given date.

    Args:
        start_date: Start of the date range (used with end_date).
        end_date: End of the date range.

    Returns:
        DataFrame with columns:
            mlb_id, stat_date, ip, w, k, walks_allowed, hits_allowed, sv, holds,
            whip (LOWEST WINS), k_bb, sv_h
    """
    boxscores = _fetch_boxscores_for_date(end_date)
    if not boxscores:
        return _empty_df(_PITCHER_STATS_COLUMNS)

    rows = _extract_pitcher_rows(boxscores, end_date)
    if not rows:
        return _empty_df(_PITCHER_STATS_COLUMNS)

    df = pd.DataFrame(rows, columns=_PITCHER_STATS_COLUMNS)
    logger.info("Loaded %d pitcher stat rows for %s.", len(df), end_date)
    return df


def get_season_stats_for_projections(
    mlb_ids: list[int],
    season: int,
) -> pd.DataFrame:
    """Fetch season-to-date stats from MLB Stats API for pace-based projections.

    Source: GET /people/{mlb_id}/stats?stats=season&group=hitting,pitching&season=YYYY

    Args:
        mlb_ids: List of MLBAM player IDs to fetch.
        season: MLB season year.

    Returns:
        DataFrame with projection columns derived from season pace.
    """
    rows: list[dict[str, Any]] = []

    for mlb_id in mlb_ids:
        url = f"{_PEOPLE_URL}/{mlb_id}/stats"
        params: dict[str, Any] = {
            "stats": "season",
            "group": "hitting,pitching",
            "sportId": 1,
            "season": season,
        }

        try:
            data = _mlb_get(url, params=params)
        except requests.RequestException as exc:
            logger.debug("Season stats failed for mlb_id=%d: %s", mlb_id, exc)
            continue

        for group in data.get("stats", []):
            splits = group.get("splits", [])
            if not splits:
                continue
            stat = splits[0].get("stat", {})
            games = stat.get("gamesPlayed", 0)
            if games == 0:
                continue

            row: dict[str, Any] = {
                "mlb_id": mlb_id,
                "fg_id": None,
                "games_played": games,
                "source": "mlb_pace",
            }

            # Batting stats
            ab = stat.get("atBats")
            if ab is not None and ab > 0:
                row.update(
                    {
                        "proj_ab": ab,
                        "proj_h": stat.get("hits", 0),
                        "proj_hr": stat.get("homeRuns", 0),
                        "proj_sb": stat.get("stolenBases", 0),
                        "proj_bb": stat.get("baseOnBalls", 0),
                        "proj_tb": stat.get("totalBases", 0),
                        "proj_avg": stat.get("avg"),
                        "proj_ops": stat.get("ops"),
                        "proj_fpct": stat.get("fielding"),
                    }
                )

            # Pitching stats
            ip_str = stat.get("inningsPitched")
            if ip_str is not None:
                try:
                    ip = float(ip_str)
                except (ValueError, TypeError):
                    ip = 0.0
                if ip > 0:
                    k = stat.get("strikeOuts", 0)
                    bb = stat.get("baseOnBalls", 0)
                    h_allowed = stat.get("hits", 0)
                    sv = stat.get("saves", 0)
                    hld = stat.get("holds", 0)
                    row.update(
                        {
                            "proj_ip": ip,
                            "proj_w": stat.get("wins", 0),
                            "proj_k": k,
                            "proj_walks_allowed": bb,
                            "proj_hits_allowed": h_allowed,
                            "proj_sv_h": sv + hld,
                            "proj_whip": stat.get("whip"),
                            "proj_k_bb": k / bb if bb > 0 else None,
                        }
                    )

            rows.append(row)

    if not rows:
        return _empty_df(_PROJECTIONS_COLUMNS)

    df = pd.DataFrame(rows)

    # Scale season stats to per-game rates, then caller multiplies by games_remaining
    for col in df.columns:
        if col.startswith("proj_") and col not in (
            "proj_avg",
            "proj_ops",
            "proj_fpct",
            "proj_whip",
            "proj_k_bb",
        ):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            games = df["games_played"].clip(lower=1)
            df[col] = df[col] / games  # Per-game rate

    df["games_remaining"] = None  # Caller fills in from schedule
    df["source"] = "mlb_pace"

    # Ensure all projection columns exist
    for col in _PROJECTIONS_COLUMNS:
        if col not in df.columns:
            df[col] = None

    result = df[_PROJECTIONS_COLUMNS].copy()
    logger.info(
        "Built %d pace-based projection rows for season %d.", len(result), season
    )
    return result


def get_steamer_projections(season: int) -> pd.DataFrame:
    """Fetch projections using MLB Stats API season stats as a pace-based proxy.

    Uses season-to-date stats from the MLB Stats API and converts them to
    per-game rates. The pipeline multiplies these rates by games_remaining
    in the current week to produce projected totals.

    NEVER raises — always returns a valid (possibly empty) DataFrame.

    Note: WHIP is lowest-wins — lower projected WHIP is better.

    Args:
        season: The MLB season year.

    Returns:
        DataFrame with columns matching fact_projections:
            mlb_id, fg_id, proj_h, proj_hr, proj_sb, proj_bb, proj_ab, proj_tb,
            proj_ip, proj_w, proj_k, proj_walks_allowed, proj_hits_allowed, proj_sv_h,
            proj_avg, proj_ops, proj_fpct, proj_whip (LOWEST WINS), proj_k_bb,
            games_remaining, source
    """
    # This function needs mlb_ids to query. When called from the pipeline,
    # dim_players should already be populated. We return empty here and let
    # the pipeline's _step_refresh_projections handle the crosswalk.
    # The actual projection data is fetched via get_season_stats_for_projections().
    logger.info(
        "get_steamer_projections called — projections now use MLB Stats API pace. "
        "Returning empty; use get_season_stats_for_projections() with mlb_ids."
    )
    return _empty_df(_PROJECTIONS_COLUMNS)


# ── Player ID crosswalk ───────────────────────────────────────────────────────


def build_player_id_crosswalk() -> pd.DataFrame:
    """Build a crosswalk table mapping player names to mlb_id and fg_id.

    Source: pybaseball.playerid_lookup() — maps between MLBAM IDs and FanGraphs IDs.

    Returns:
        DataFrame with columns: full_name, mlb_id, fg_id
        Used during integration to populate dim_players.mlb_id and dim_players.fg_id.
        Never blocks on failure — returns empty DataFrame with correct columns on error.
    """
    import pybaseball  # noqa: PLC0415

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # playerid_lookup with no args returns the full Chadwick register
            raw = pybaseball.chadwick_register()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "pybaseball.chadwick_register() failed: %s. Returning empty crosswalk.", exc
        )
        return _empty_df(_CROSSWALK_COLUMNS)

    if raw is None or raw.empty:
        logger.warning("pybaseball.chadwick_register() returned no data.")
        return _empty_df(_CROSSWALK_COLUMNS)

    # Chadwick register columns: key_mlbam, key_fangraphs, name_first, name_last
    col_map = {
        "key_mlbam": "mlb_id",
        "key_fangraphs": "fg_id",
    }
    existing = {k: v for k, v in col_map.items() if k in raw.columns}
    df = raw.rename(columns=existing).copy()

    # Build full_name from first + last
    if "name_first" in df.columns and "name_last" in df.columns:
        df["full_name"] = (
            df["name_first"].fillna("").str.strip()
            + " "
            + df["name_last"].fillna("").str.strip()
        ).str.strip()
    else:
        df["full_name"] = ""

    for col in _CROSSWALK_COLUMNS:
        if col not in df.columns:
            df[col] = None

    result: pd.DataFrame = df[_CROSSWALK_COLUMNS].dropna(subset=["mlb_id"]).copy()
    logger.info("Built player ID crosswalk with %d entries.", len(result))
    return result
