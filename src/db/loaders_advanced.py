"""
loaders_advanced.py

MotherDuck loader for fact_player_advanced_stats.

This loader fuses two sources per season:
  1. Baseball Savant leaderboards (via src.api.mlb_client.get_savant_*_advanced)
     keyed on MLBAM mlb_id.
  2. Season-to-date raw components from fact_player_stats_daily, used to compute
     wOBA (batters), FIP and K-BB% (pitchers).

Savant rows are joined to Yahoo player_id via dim_players.mlb_id. Rows with no
matching player in dim_players are dropped.

The final table uses INSERT OR REPLACE on (player_id, season).
"""

from __future__ import annotations

import datetime
import logging

import duckdb
import pandas as pd

from src.db.schema import (
    DIM_PLAYERS,
    FACT_PLAYER_ADVANCED_STATS,
    FACT_PLAYER_STATS_DAILY,
)

logger = logging.getLogger(__name__)

# ── wOBA / FIP constants ─────────────────────────────────────────────────────
#
# Using standard FanGraphs linear weights (values vary slightly by season;
# these are the commonly-cited 2024 guide values and are accurate enough for
# a roster UI display).

_WOBA_WEIGHTS = {
    "ubb": 0.696,  # unintentional walk
    "hbp": 0.727,
    "1b": 0.887,
    "2b": 1.250,
    "3b": 1.583,
    "hr": 2.031,
}

_ADVANCED_COLUMNS = [
    "player_id",
    "season",
    "xwoba",
    "woba",
    "barrel_pct",
    "hard_hit_pct",
    "avg_launch_angle",
    "sweet_spot_pct",
    "bat_speed_pctile",
    "sprint_speed_pctile",
    "xera",
    "xwoba_against",
    "k_bb_pct",
    "barrel_pct_against",
    "hard_hit_pct_against",
    "updated_at",
]


def _compute_batter_derived(
    conn: duckdb.DuckDBPyConnection, season: int
) -> pd.DataFrame:
    """Compute season-to-date wOBA per player from fact_player_stats_daily.

    Uses raw components: h, hr, 2B and 3B are derived from tb = 1B+2·2B+3·3B+4·HR
    and h. Since the daily table does not separate singles/doubles/triples, we
    approximate using tb and hr:
      1B ≈ h - (extra-base hits). We store tb and h, so we can only recover an
      approximation of extra-base distribution. To keep this simple and avoid
      misleading precision, we compute wOBA using this approximation:
        singles  = h - hr - doubles - triples  (doubles/triples unknown)
      We collapse to a workable form by treating all non-HR hits as singles
      weighted at 0.887 and distributing the residual tb above (h + 3·hr) as
      extra-base value at the average weight of 1.25 (doubles weight).
    """
    rows = conn.execute(
        f"""
        SELECT
            player_id,
            SUM(COALESCE(h, 0))              AS h,
            SUM(COALESCE(hr, 0))             AS hr,
            SUM(COALESCE(bb, 0))             AS bb,
            SUM(COALESCE(hbp, 0))            AS hbp,
            SUM(COALESCE(ab, 0))             AS ab,
            SUM(COALESCE(sf, 0))             AS sf,
            SUM(COALESCE(tb, 0))             AS tb
        FROM {FACT_PLAYER_STATS_DAILY}
        WHERE EXTRACT(year FROM stat_date) = {season}
        GROUP BY player_id
        HAVING SUM(COALESCE(ab, 0)) > 0
        """
    ).fetchdf()

    if rows.empty:
        return pd.DataFrame(columns=["player_id", "season", "woba"])

    denom = rows["ab"] + rows["bb"] + rows["sf"] + rows["hbp"]
    singles = (rows["h"] - rows["hr"]).clip(lower=0)
    # Extra-base bases above the singles contribution: tb − (1·1B + 4·HR)
    # Distribute these as "double-equivalent" bases at weight 1.25.
    xb_bases = (rows["tb"] - singles - 4 * rows["hr"]).clip(lower=0)
    numer = (
        _WOBA_WEIGHTS["ubb"] * rows["bb"]
        + _WOBA_WEIGHTS["hbp"] * rows["hbp"]
        + _WOBA_WEIGHTS["1b"] * singles
        + _WOBA_WEIGHTS["2b"] * xb_bases  # approximated as doubles
        + _WOBA_WEIGHTS["hr"] * rows["hr"]
    )
    rows["woba"] = (numer / denom).round(3)
    rows["season"] = season
    return rows[["player_id", "season", "woba"]]


def _compute_pitcher_derived(
    conn: duckdb.DuckDBPyConnection, season: int
) -> pd.DataFrame:
    """Compute season-to-date K-BB% per pitcher.

    K-BB%  = (K − BB) / BF × 100     (BF approximated as outs + H + BB)

    Note: FIP cannot be computed because fact_player_stats_daily does not track
    HR allowed by pitchers (only HR hit by batters). FIP is left NULL in the
    fact_player_advanced_stats table.
    """
    rows = conn.execute(
        f"""
        SELECT
            player_id,
            SUM(COALESCE(ip, 0))             AS ip,
            SUM(COALESCE(k, 0))              AS k,
            SUM(COALESCE(walks_allowed, 0))  AS walks_allowed,
            SUM(COALESCE(hits_allowed, 0))   AS hits_allowed
        FROM {FACT_PLAYER_STATS_DAILY}
        WHERE EXTRACT(year FROM stat_date) = {season}
        GROUP BY player_id
        HAVING SUM(COALESCE(ip, 0)) > 0
        """
    ).fetchdf()

    if rows.empty:
        return pd.DataFrame(columns=["player_id", "season", "k_bb_pct"])

    ip = rows["ip"].astype(float)
    # Approximate batters faced = outs (IP·3) + hits allowed + walks allowed
    bf = ip * 3 + rows["hits_allowed"] + rows["walks_allowed"]
    k_minus_bb = rows["k"] - rows["walks_allowed"]
    rows["k_bb_pct"] = ((k_minus_bb / bf) * 100).round(2)
    rows["season"] = season
    return rows[["player_id", "season", "k_bb_pct"]]


def _map_mlb_to_yahoo(
    conn: duckdb.DuckDBPyConnection, df: pd.DataFrame
) -> pd.DataFrame:
    """Replace mlb_id with Yahoo player_id via dim_players."""
    if df.empty or "mlb_id" not in df.columns:
        return df
    crosswalk = conn.execute(
        f"SELECT player_id, mlb_id FROM {DIM_PLAYERS} WHERE mlb_id IS NOT NULL"
    ).fetchdf()
    if crosswalk.empty:
        return df.iloc[0:0]
    crosswalk["mlb_id"] = pd.to_numeric(crosswalk["mlb_id"], errors="coerce").astype(
        "Int64"
    )
    df = df.copy()
    df["mlb_id"] = pd.to_numeric(df["mlb_id"], errors="coerce").astype("Int64")
    merged = df.merge(crosswalk, on="mlb_id", how="inner")
    return merged.drop(columns=["mlb_id"])


def load_advanced_stats(
    conn: duckdb.DuckDBPyConnection,
    season: int,
    savant_batter: pd.DataFrame,
    savant_pitcher: pd.DataFrame,
) -> int:
    """Merge Savant + computed advanced stats and upsert into MotherDuck.

    Args:
        conn: Open DuckDB connection.
        season: MLB season year.
        savant_batter: Output of mlb_client.get_savant_batter_advanced(season).
        savant_pitcher: Output of mlb_client.get_savant_pitcher_advanced(season).

    Returns:
        Number of rows upserted.
    """
    batter_savant = _map_mlb_to_yahoo(conn, savant_batter)
    pitcher_savant = _map_mlb_to_yahoo(conn, savant_pitcher)
    batter_derived = _compute_batter_derived(conn, season)
    pitcher_derived = _compute_pitcher_derived(conn, season)

    # Outer-merge all four frames on player_id.
    frames = [
        f
        for f in (batter_savant, pitcher_savant, batter_derived, pitcher_derived)
        if not f.empty
    ]
    if not frames:
        logger.info("load_advanced_stats: nothing to load for season=%s.", season)
        return 0

    merged: pd.DataFrame | None = None
    for f in frames:
        f = f.drop(columns=["season"], errors="ignore")
        merged = f if merged is None else merged.merge(f, on="player_id", how="outer")

    assert merged is not None
    merged["season"] = season
    merged["updated_at"] = datetime.datetime.now(datetime.UTC)

    for col in _ADVANCED_COLUMNS:
        if col not in merged.columns:
            merged[col] = None

    insert_df = merged[_ADVANCED_COLUMNS].dropna(subset=["player_id"]).copy()
    if insert_df.empty:
        logger.info("load_advanced_stats: no matchable players for season=%s.", season)
        return 0

    conn.register("_advanced_stats_staging", insert_df)
    conn.execute(
        f"INSERT OR REPLACE INTO {FACT_PLAYER_ADVANCED_STATS} "
        "SELECT * FROM _advanced_stats_staging"
    )
    conn.unregister("_advanced_stats_staging")

    logger.info(
        "load_advanced_stats: upserted %d rows for season=%s.",
        len(insert_df),
        season,
    )
    return len(insert_df)
