"""
loaders_mlb.py

MotherDuck loaders for MLB-sourced data.

This module is intentionally separate from loaders.py (Yahoo data) to avoid
merge conflicts when both agents are working in parallel.

All upserts use INSERT OR REPLACE.
All rate stats in weekly aggregates are computed from raw components:
  - avg  = SUM(h) / SUM(ab)              — never AVG(avg)
  - whip = (SUM(walks_allowed) + SUM(hits_allowed)) / SUM(ip)  [LOWEST WINS]
  - ops  computed from SUM(tb)/SUM(ab) + OBP components
  - k_bb = SUM(k) / SUM(walks_allowed)  — None if SUM(walks_allowed) == 0
  - fpct = (SUM(chances) - SUM(errors)) / SUM(chances)
"""

from __future__ import annotations

import datetime
import logging

import duckdb
import pandas as pd

from src.db.schema import (
    DIM_DATES,
    DIM_PLAYERS,
    FACT_PLAYER_STATS_DAILY,
    FACT_PLAYER_STATS_WEEKLY,
    FACT_PROJECTIONS,
)

logger = logging.getLogger(__name__)

# ── Required column sets for validation ───────────────────────────────────────

_DAILY_STATS_REQUIRED = {"player_id", "stat_date"}

_WEEKLY_STATS_REQUIRED = {"player_id", "week_number", "season"}

_PROJECTIONS_REQUIRED = {"player_id", "projection_date", "target_week", "source"}

_DIM_DATES_COLUMNS = ["date", "season", "week_number", "is_playoff_week", "day_of_week"]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _validate_columns(df: pd.DataFrame, required: set[str], loader_name: str) -> None:
    """Raise ValueError if any required columns are absent from the DataFrame.

    Args:
        df: The DataFrame to validate.
        required: Set of required column names.
        loader_name: Name of the calling loader (for error messages).

    Raises:
        ValueError: If any required column is missing.
    """
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{loader_name}: DataFrame is missing required columns: {sorted(missing)}"
        )


def get_fantasy_week(date: datetime.date, season_start: datetime.date) -> int:
    """Compute the fantasy week number for a given date relative to season start.

    Week 1 starts on the first Monday on or before the season start date.
    Each subsequent week begins on Monday.

    Args:
        date: The date to compute the week for.
        season_start: The first day of the MLB regular season.

    Returns:
        Integer week number (1-indexed). Returns 1 if date is before season_start.
    """
    # Find the Monday on or before season_start (week 1 anchor)
    week1_start = season_start - datetime.timedelta(days=season_start.weekday())
    delta_days = (date - week1_start).days
    if delta_days < 0:
        return 1
    return (delta_days // 7) + 1


# Season start dates by year (first day of MLB regular season)
_SEASON_STARTS: dict[int, datetime.date] = {
    2024: datetime.date(2024, 3, 20),
    2025: datetime.date(2025, 3, 27),
    2026: datetime.date(2026, 3, 26),
}


def _get_season_start(season: int) -> datetime.date:
    """Return the season start date, defaulting to March 27 if unknown."""
    return _SEASON_STARTS.get(season, datetime.date(season, 3, 27))


# ── Loaders ───────────────────────────────────────────────────────────────────


def load_daily_stats(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    stat_date: datetime.date,
) -> int:
    """Upsert daily player stats into fact_player_stats_daily.

    Args:
        conn: Open DuckDB connection.
        df: DataFrame with player stats. Must include columns: player_id, stat_date.
        stat_date: The date these stats represent (used as a fallback if stat_date
                   column is missing or to confirm consistency).

    Returns:
        Number of rows upserted.

    Raises:
        ValueError: If required columns are missing.
    """
    _validate_columns(df, _DAILY_STATS_REQUIRED, "load_daily_stats")

    if df.empty:
        logger.info("load_daily_stats: empty DataFrame for %s, skipping.", stat_date)
        return 0

    # Ensure stat_date column is set to the provided date
    working = df.copy()
    working["stat_date"] = stat_date

    # All columns in fact_player_stats_daily
    table_columns = [
        "player_id",
        "stat_date",
        "ab",
        "h",
        "hr",
        "sb",
        "bb",
        "hbp",
        "sf",
        "tb",
        "errors",
        "chances",
        "ip",
        "w",
        "k",
        "walks_allowed",
        "hits_allowed",
        "sv",
        "holds",
        "avg",
        "ops",
        "fpct",
        "whip",  # LOWEST WINS — lower is better
        "k_bb",
        "sv_h",
    ]

    # Add missing columns as NULL
    for col in table_columns:
        if col not in working.columns:
            working[col] = None

    insert_df = working[table_columns].copy()

    conn.register("_daily_stats_staging", insert_df)
    conn.execute(
        f"INSERT OR REPLACE INTO {FACT_PLAYER_STATS_DAILY} SELECT * FROM _daily_stats_staging"
    )
    conn.unregister("_daily_stats_staging")

    row_count = len(insert_df)
    logger.info(
        "load_daily_stats: upserted %d rows for stat_date=%s.", row_count, stat_date
    )
    return row_count


def load_weekly_stats(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    week_number: int,
    season: int,
) -> int:
    """Rebuild weekly stats for a given week from daily aggregates.

    Rate stats are computed from aggregated raw components — never averaged:
      - avg  = SUM(h) / SUM(ab)
      - whip = (SUM(walks_allowed) + SUM(hits_allowed)) / SUM(ip)  [LOWEST WINS]
      - ops  = SLG + OBP = SUM(tb)/SUM(ab) + (SUM(h)+SUM(bb)+SUM(hbp)) /
               (SUM(ab)+SUM(bb)+SUM(hbp)+SUM(sf))
      - k_bb = SUM(k) / SUM(walks_allowed)  — None if SUM(walks_allowed) == 0
      - fpct = (SUM(chances) - SUM(errors)) / SUM(chances)
      - sv_h = SUM(sv) + SUM(holds)

    Args:
        conn: Open DuckDB connection.
        df: DataFrame with daily player stats. Must include player_id, week_number,
            season columns (or they are derived). If df is empty, reads from
            fact_player_stats_daily directly.
        week_number: Fantasy week number to rebuild.
        season: MLB season year.

    Returns:
        Number of rows written.

    Raises:
        ValueError: If required columns are missing in df (when df is non-empty).
    """
    if not df.empty:
        _validate_columns(df, _WEEKLY_STATS_REQUIRED, "load_weekly_stats")

    # Delete existing rows for this week/season before rebuilding
    conn.execute(
        f"DELETE FROM {FACT_PLAYER_STATS_WEEKLY} "
        f"WHERE week_number = {week_number} AND season = {season}"
    )

    if df.empty:
        logger.info(
            "load_weekly_stats: empty DataFrame for week %d season %d, skipping.",
            week_number,
            season,
        )
        return 0

    # Aggregate raw components per player
    agg_cols = {
        "ab": "sum",
        "h": "sum",
        "hr": "sum",
        "sb": "sum",
        "bb": "sum",
        "hbp": "sum",
        "sf": "sum",
        "tb": "sum",
        "errors": "sum",
        "chances": "sum",
        "ip": "sum",
        "w": "sum",
        "k": "sum",
        "walks_allowed": "sum",
        "hits_allowed": "sum",
        "sv": "sum",
        "holds": "sum",
    }

    present_agg = {col: agg for col, agg in agg_cols.items() if col in df.columns}
    grouped = df.groupby("player_id").agg(present_agg).reset_index()

    # Ensure all raw component columns exist (fill with 0 for safe arithmetic)
    for col in agg_cols:
        if col not in grouped.columns:
            grouped[col] = 0

    grouped["week_number"] = week_number
    grouped["season"] = season

    # ── Compute rate stats from components ────────────────────────────────────

    # avg = SUM(h) / SUM(ab)
    ab = pd.to_numeric(grouped["ab"], errors="coerce")
    h = pd.to_numeric(grouped["h"], errors="coerce")
    grouped["avg"] = h.where(ab > 0).div(ab.where(ab > 0))

    # ops = SLG + OBP
    # SLG = SUM(tb) / SUM(ab)
    # OBP = (SUM(h) + SUM(bb) + SUM(hbp)) / (SUM(ab) + SUM(bb) + SUM(hbp) + SUM(sf))
    tb = pd.to_numeric(grouped["tb"], errors="coerce")
    bb = pd.to_numeric(grouped["bb"], errors="coerce")
    hbp = pd.to_numeric(grouped["hbp"], errors="coerce")
    sf = pd.to_numeric(grouped["sf"], errors="coerce")
    slg = tb.where(ab > 0).div(ab.where(ab > 0))
    obp_num = h + bb + hbp
    obp_denom = ab + bb + hbp + sf
    obp = obp_num.where(obp_denom > 0).div(obp_denom.where(obp_denom > 0))
    grouped["ops"] = slg + obp

    # fpct = (SUM(chances) - SUM(errors)) / SUM(chances)
    chances = pd.to_numeric(grouped["chances"], errors="coerce")
    errors = pd.to_numeric(grouped["errors"], errors="coerce")
    grouped["fpct"] = (
        (chances - errors).where(chances > 0).div(chances.where(chances > 0))
    )

    # whip = (SUM(walks_allowed) + SUM(hits_allowed)) / SUM(ip)  [LOWEST WINS]
    ip = pd.to_numeric(grouped["ip"], errors="coerce")
    walks_allowed = pd.to_numeric(grouped["walks_allowed"], errors="coerce")
    hits_allowed = pd.to_numeric(grouped["hits_allowed"], errors="coerce")
    grouped["whip"] = (walks_allowed + hits_allowed).where(ip > 0).div(ip.where(ip > 0))

    # k_bb = SUM(k) / SUM(walks_allowed) — None if walks_allowed == 0
    k = pd.to_numeric(grouped["k"], errors="coerce")
    grouped["k_bb"] = k.where(walks_allowed > 0).div(
        walks_allowed.where(walks_allowed > 0)
    )

    # sv_h = SUM(sv) + SUM(holds)
    sv = pd.to_numeric(grouped["sv"], errors="coerce").fillna(0)
    holds = pd.to_numeric(grouped["holds"], errors="coerce").fillna(0)
    grouped["sv_h"] = (sv + holds).astype(int)

    # ── Write to table ────────────────────────────────────────────────────────
    table_columns = [
        "player_id",
        "week_number",
        "season",
        "ab",
        "h",
        "hr",
        "sb",
        "bb",
        "hbp",
        "sf",
        "tb",
        "errors",
        "chances",
        "ip",
        "w",
        "k",
        "walks_allowed",
        "hits_allowed",
        "sv",
        "holds",
        "avg",
        "ops",
        "fpct",
        "whip",  # LOWEST WINS
        "k_bb",
        "sv_h",
    ]

    for col in table_columns:
        if col not in grouped.columns:
            grouped[col] = None

    insert_df = grouped[table_columns].copy()

    conn.register("_weekly_stats_staging", insert_df)
    conn.execute(
        f"INSERT OR REPLACE INTO {FACT_PLAYER_STATS_WEEKLY} "
        "SELECT * FROM _weekly_stats_staging"
    )
    conn.unregister("_weekly_stats_staging")

    row_count = len(insert_df)
    logger.info(
        "load_weekly_stats: wrote %d rows for week %d season %d.",
        row_count,
        week_number,
        season,
    )
    return row_count


def load_projections(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
) -> int:
    """Upsert projections into fact_projections.

    Args:
        conn: Open DuckDB connection.
        df: DataFrame with projection data. Must include columns:
            player_id, projection_date, target_week, source.

    Returns:
        Number of rows upserted.

    Raises:
        ValueError: If required columns are missing.
    """
    _validate_columns(df, _PROJECTIONS_REQUIRED, "load_projections")

    if df.empty:
        logger.info("load_projections: empty DataFrame, skipping.")
        return 0

    table_columns = [
        "player_id",
        "projection_date",
        "target_week",
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

    working = df.copy()
    for col in table_columns:
        if col not in working.columns:
            working[col] = None

    insert_df = working[table_columns].copy()

    conn.register("_projections_staging", insert_df)
    conn.execute(
        f"INSERT OR REPLACE INTO {FACT_PROJECTIONS} SELECT * FROM _projections_staging"
    )
    conn.unregister("_projections_staging")

    row_count = len(insert_df)
    logger.info("load_projections: upserted %d rows.", row_count)
    return row_count


def load_dim_dates(
    conn: duckdb.DuckDBPyConnection,
    season: int,
    start_date: datetime.date,
    end_date: datetime.date,
) -> int:
    """Populate dim_dates for the given date range.

    Generates one row per calendar date from start_date to end_date inclusive.
    week_number is derived using get_fantasy_week() relative to the season start.
    is_playoff_week is not set here (defaults to false — update separately if needed).

    Args:
        conn: Open DuckDB connection.
        season: MLB season year (used for week_number computation).
        start_date: First date in range (inclusive).
        end_date: Last date in range (inclusive).

    Returns:
        Number of rows upserted.
    """
    if start_date > end_date:
        logger.warning(
            "load_dim_dates: start_date %s > end_date %s, nothing to insert.",
            start_date,
            end_date,
        )
        return 0

    season_start = _get_season_start(season)
    rows: list[dict[str, object]] = []

    current = start_date
    while current <= end_date:
        week_num = get_fantasy_week(current, season_start)
        rows.append(
            {
                "date": current,
                "season": season,
                "week_number": week_num,
                "is_playoff_week": False,
                "day_of_week": current.strftime("%A"),
            }
        )
        current += datetime.timedelta(days=1)

    insert_df = pd.DataFrame(rows, columns=_DIM_DATES_COLUMNS)

    conn.register("_dim_dates_staging", insert_df)
    conn.execute(f"INSERT OR REPLACE INTO {DIM_DATES} SELECT * FROM _dim_dates_staging")
    conn.unregister("_dim_dates_staging")

    row_count = len(insert_df)
    logger.info(
        "load_dim_dates: upserted %d rows for season %d (%s to %s).",
        row_count,
        season,
        start_date,
        end_date,
    )
    return row_count


def update_player_crosswalk(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
) -> int:
    """Update mlb_id and fg_id on existing dim_players rows.

    Matches by full_name (case-insensitive). Only updates rows that already
    exist in dim_players — does not insert new players.

    Args:
        conn: Open DuckDB connection.
        df: DataFrame with columns: full_name, mlb_id, fg_id.

    Returns:
        Number of rows updated.

    Raises:
        ValueError: If required columns (full_name) are missing.
    """
    _validate_columns(df, {"full_name"}, "update_player_crosswalk")

    if df.empty:
        logger.info("update_player_crosswalk: empty DataFrame, skipping.")
        return 0

    working = df.copy()
    if "mlb_id" not in working.columns:
        working["mlb_id"] = None
    if "fg_id" not in working.columns:
        working["fg_id"] = None

    crosswalk_df = working[["full_name", "mlb_id", "fg_id"]].copy()
    conn.register("_crosswalk_staging", crosswalk_df)

    # Update existing dim_players rows where full_name matches
    conn.execute(f"""
        UPDATE {DIM_PLAYERS}
        SET
            mlb_id = s.mlb_id,
            fg_id  = s.fg_id
        FROM _crosswalk_staging s
        WHERE LOWER({DIM_PLAYERS}.full_name) = LOWER(s.full_name)
          AND s.mlb_id IS NOT NULL
    """)

    conn.unregister("_crosswalk_staging")

    # DuckDB doesn't return updated row counts directly from UPDATE — report input size
    row_count = len(crosswalk_df.dropna(subset=["mlb_id"]))
    logger.info("update_player_crosswalk: processed %d crosswalk entries.", row_count)
    return row_count
