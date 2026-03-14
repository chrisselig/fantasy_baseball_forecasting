"""
mlb_client.py

MLB data ingestion client for the fantasy baseball pipeline.

Covers:
  - MLB Stats API (public, no auth): call-ups, player info, game schedule,
    minor league stats
  - pybaseball integration: batter/pitcher stats, Steamer projections
  - Player ID crosswalk builder (MLBAM ↔ FanGraphs)

All pybaseball calls are wrapped to never block the pipeline — failures return
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

# Columns matching fact_player_stats_daily (batter portion)
_BATTER_STATS_COLUMNS = [
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


# ── pybaseball integration ────────────────────────────────────────────────────


def get_batter_stats(
    start_date: datetime.date,
    end_date: datetime.date,
) -> pd.DataFrame:
    """Fetch batter stats via pybaseball.

    Source: pybaseball.batting_stats() for season-level.
    Normalize to fact_player_stats_daily columns.
    Cache results: raise a warning and return empty DataFrame if pybaseball fails
    (never block the pipeline on a pybaseball failure).

    Args:
        start_date: Start of the date range (used to derive the season year).
        end_date: End of the date range.

    Returns:
        DataFrame with columns matching fact_player_stats_daily:
            player_id (None — caller maps via dim_players.mlb_id),
            mlb_id, stat_date, ab, h, hr, sb, bb, hbp, sf, tb,
            avg, ops, fpct, errors, chances
    """
    import pybaseball  # noqa: PLC0415

    season = start_date.year
    stat_date = end_date  # Season-level stats tagged to the end_date

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = pybaseball.batting_stats(season)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "pybaseball.batting_stats() failed: %s. Returning empty DataFrame.", exc
        )
        return _empty_df(_BATTER_STATS_COLUMNS)

    if raw is None or raw.empty:
        logger.warning(
            "pybaseball.batting_stats() returned no data for season %d.", season
        )
        return _empty_df(_BATTER_STATS_COLUMNS)

    # Build crosswalk: pybaseball batting_stats includes an 'IDfg' column (FanGraphs ID)
    # but not directly an mlb_id. We normalise what we can and leave mlb_id resolution
    # to the crosswalk step.
    col_map = {
        "IDfg": "fg_id",
        "AB": "ab",
        "H": "h",
        "HR": "hr",
        "SB": "sb",
        "BB": "bb",
        "HBP": "hbp",
        "SF": "sf",
        "AVG": "avg",
        "OPS": "ops",
    }
    existing_cols = {k: v for k, v in col_map.items() if k in raw.columns}
    df = raw.rename(columns=existing_cols).copy()

    # Add missing columns with None
    for col in _BATTER_STATS_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df["player_id"] = None
    df["mlb_id"] = None  # caller resolves via crosswalk
    df["stat_date"] = stat_date

    # Compute total bases if not present (requires 1B, 2B, 3B, HR — approximate from hits)
    if "tb" not in raw.columns or df["tb"].isna().all():
        tb_cols_present = all(c in raw.columns for c in ["H", "2B", "3B", "HR"])
        if tb_cols_present:
            singles = raw["H"] - raw.get("2B", 0) - raw.get("3B", 0) - raw["HR"]
            df["tb"] = (
                singles + 2 * raw.get("2B", 0) + 3 * raw.get("3B", 0) + 4 * raw["HR"]
            )

    # fpct, errors, chances: not available from batting_stats — leave as None
    df["fpct"] = None
    df["errors"] = None
    df["chances"] = None

    result: pd.DataFrame = df[_BATTER_STATS_COLUMNS].copy()
    logger.info("Loaded %d batter stat rows for season %d.", len(result), season)
    return result


def get_pitcher_stats(
    start_date: datetime.date,
    end_date: datetime.date,
) -> pd.DataFrame:
    """Fetch pitcher stats via pybaseball.

    Source: pybaseball.pitching_stats() for season-level.
    Normalize to fact_player_stats_daily columns.
    Never block on failure — return empty DataFrame with correct columns.

    Args:
        start_date: Start of the date range (used to derive the season year).
        end_date: End of the date range.

    Returns:
        DataFrame with columns:
            mlb_id, stat_date, ip, w, k, walks_allowed, hits_allowed, sv, holds,
            whip (LOWEST WINS), k_bb, sv_h
    """
    import pybaseball  # noqa: PLC0415

    season = start_date.year
    stat_date = end_date

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = pybaseball.pitching_stats(season)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "pybaseball.pitching_stats() failed: %s. Returning empty DataFrame.", exc
        )
        return _empty_df(_PITCHER_STATS_COLUMNS)

    if raw is None or raw.empty:
        logger.warning(
            "pybaseball.pitching_stats() returned no data for season %d.", season
        )
        return _empty_df(_PITCHER_STATS_COLUMNS)

    col_map = {
        "IP": "ip",
        "W": "w",
        "SO": "k",
        "BB": "walks_allowed",
        "H": "hits_allowed",
        "SV": "sv",
        "HLD": "holds",
        "WHIP": "whip",  # LOWEST WINS
    }
    existing_cols = {k: v for k, v in col_map.items() if k in raw.columns}
    df = raw.rename(columns=existing_cols).copy()

    df["mlb_id"] = None  # caller resolves via crosswalk
    df["stat_date"] = stat_date

    # Compute k_bb = k / walks_allowed (handle division by zero → None)
    if "k" in df.columns and "walks_allowed" in df.columns:
        denom = pd.to_numeric(df["walks_allowed"], errors="coerce")
        num = pd.to_numeric(df["k"], errors="coerce")
        df["k_bb"] = num.where(denom > 0).div(denom.where(denom > 0))
    else:
        df["k_bb"] = None

    # sv_h = sv + holds
    sv_col = pd.to_numeric(df.get("sv", 0), errors="coerce").fillna(0)
    holds_col = pd.to_numeric(df.get("holds", 0), errors="coerce").fillna(0)
    df["sv_h"] = (sv_col + holds_col).astype(int)

    for col in _PITCHER_STATS_COLUMNS:
        if col not in df.columns:
            df[col] = None

    result: pd.DataFrame = df[_PITCHER_STATS_COLUMNS].copy()
    logger.info("Loaded %d pitcher stat rows for season %d.", len(result), season)
    return result


def get_steamer_projections(season: int) -> pd.DataFrame:
    """Fetch Steamer projections via pybaseball.

    Primary: pybaseball.fg_batting_projections() + pybaseball.fg_pitching_projections()
    Fallback: return empty DataFrame with correct columns and log a WARNING.
    NEVER raise — always return a valid (possibly empty) DataFrame.
    Set source='steamer' on success, source='unavailable' on fallback.

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
    import pybaseball  # noqa: PLC0415

    rows: list[pd.DataFrame] = []

    # --- Batting projections ---
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bat_raw = pybaseball.fg_batting_projections(season)

        if bat_raw is not None and not bat_raw.empty:
            bat_col_map = {
                "playerid": "fg_id",
                "H": "proj_h",
                "HR": "proj_hr",
                "SB": "proj_sb",
                "BB": "proj_bb",
                "AB": "proj_ab",
                "AVG": "proj_avg",
                "OPS": "proj_ops",
            }
            existing = {k: v for k, v in bat_col_map.items() if k in bat_raw.columns}
            bat_df = bat_raw.rename(columns=existing).copy()
            bat_df["source"] = "steamer"
            bat_df["mlb_id"] = None  # resolved via crosswalk
            # Estimated total bases from AB and OPS components (best effort)
            bat_df["proj_tb"] = None
            bat_df["proj_fpct"] = None
            bat_df["proj_ip"] = None
            bat_df["proj_w"] = None
            bat_df["proj_k"] = None
            bat_df["proj_walks_allowed"] = None
            bat_df["proj_hits_allowed"] = None
            bat_df["proj_sv_h"] = None
            bat_df["proj_whip"] = None  # LOWEST WINS — not applicable to batters
            bat_df["proj_k_bb"] = None
            bat_df["games_remaining"] = None
            rows.append(bat_df)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "pybaseball.fg_batting_projections() failed: %s. Continuing to pitching.",
            exc,
        )

    # --- Pitching projections ---
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pit_raw = pybaseball.fg_pitching_projections(season)

        if pit_raw is not None and not pit_raw.empty:
            pit_col_map = {
                "playerid": "fg_id",
                "IP": "proj_ip",
                "W": "proj_w",
                "SO": "proj_k",
                "BB": "proj_walks_allowed",
                "H": "proj_hits_allowed",
                "SV": "proj_sv",
                "HLD": "proj_hld",
                "WHIP": "proj_whip",  # LOWEST WINS
            }
            existing = {k: v for k, v in pit_col_map.items() if k in pit_raw.columns}
            pit_df = pit_raw.rename(columns=existing).copy()
            pit_df["source"] = "steamer"
            pit_df["mlb_id"] = None

            # sv_h = sv + holds
            sv_col = pd.to_numeric(pit_df.get("proj_sv", 0), errors="coerce").fillna(0)
            hld_col = pd.to_numeric(pit_df.get("proj_hld", 0), errors="coerce").fillna(
                0
            )
            pit_df["proj_sv_h"] = (sv_col + hld_col).astype(int)

            # k_bb
            if "proj_k" in pit_df.columns and "proj_walks_allowed" in pit_df.columns:
                denom = pd.to_numeric(pit_df["proj_walks_allowed"], errors="coerce")
                num = pd.to_numeric(pit_df["proj_k"], errors="coerce")
                pit_df["proj_k_bb"] = num.where(denom > 0).div(denom.where(denom > 0))
            else:
                pit_df["proj_k_bb"] = None

            pit_df["proj_h"] = None
            pit_df["proj_hr"] = None
            pit_df["proj_sb"] = None
            pit_df["proj_bb"] = None
            pit_df["proj_ab"] = None
            pit_df["proj_tb"] = None
            pit_df["proj_avg"] = None
            pit_df["proj_ops"] = None
            pit_df["proj_fpct"] = None
            pit_df["games_remaining"] = None
            rows.append(pit_df)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "pybaseball.fg_pitching_projections() failed: %s. Continuing.", exc
        )

    if not rows:
        logger.warning(
            "All Steamer projection sources failed for season %d. "
            "Returning empty DataFrame with source='unavailable'.",
            season,
        )
        df = _empty_df(_PROJECTIONS_COLUMNS)
        return df

    combined = pd.concat(rows, ignore_index=True)
    for col in _PROJECTIONS_COLUMNS:
        if col not in combined.columns:
            combined[col] = None

    result = combined[_PROJECTIONS_COLUMNS].copy()
    logger.info("Loaded %d Steamer projection rows for season %d.", len(result), season)
    return result


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
