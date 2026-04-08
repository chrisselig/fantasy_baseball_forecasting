"""
daily_run.py

GitHub Actions daily pipeline entry point.

Orchestrates the full ETL + analysis cycle:
  1. Ensure schema exists
  2. Load Yahoo Fantasy data (roster, transactions, free agents, players)
  3. Load MLB stats (daily stats, schedule, callups)
  4. Refresh Steamer projections
  5. Run analysis (matchup scoring, waiver ranking, lineup optimization)
  6. Write daily report to fact_daily_reports
  7. Log pipeline run to fact_pipeline_runs

Usage::

    from src.pipeline.daily_run import run_daily_pipeline
    from src.db.connection import managed_connection
    from src.config import load_league_settings

    settings = load_league_settings()
    with managed_connection() as conn:
        result = run_daily_pipeline(conn, settings)
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import uuid
from typing import Any

import duckdb
import pandas as pd

from src.analysis.lineup_optimizer import (
    build_daily_report,
    optimize_daily_lineup,
    recommend_adds,
)
from src.analysis.matchup_analyzer import (
    check_ip_pace,
    project_week_totals,
    score_categories,
)
from src.analysis.waiver_ranker import rank_free_agents
from src.api import mlb_client
from src.api.yahoo_client import YahooClient
from src.config import LeagueSettings
from src.db.loaders_mlb import (
    get_fantasy_week,
    load_daily_stats,
    load_projections,
    update_player_crosswalk,
)
from src.db.loaders_yahoo import (
    load_matchups,
    load_players,
    load_rosters,
    load_transactions,
    stage_free_agents,
)
from src.db.schema import (
    DIM_PLAYERS,
    FACT_DAILY_REPORTS,
    FACT_MATCHUPS,
    FACT_PIPELINE_RUNS,
    FACT_PLAYER_STATS_DAILY,
    FACT_PROJECTIONS,
    FACT_ROSTERS,
    FACT_TRANSACTIONS,
    FACT_WAIVER_SCORES,
    create_all_tables,
)

logger = logging.getLogger(__name__)

# MLB season start dates (first day of regular season)
_SEASON_STARTS: dict[int, datetime.date] = {
    2024: datetime.date(2024, 3, 20),
    2025: datetime.date(2025, 3, 27),
    2026: datetime.date(2026, 3, 26),
}

# Static MLB team ID → abbreviation mapping (used to build player-level schedule)
_MLB_TEAM_ID_TO_ABBR: dict[int, str] = {
    108: "LAA",
    109: "ARI",
    110: "BAL",
    111: "BOS",
    112: "CHC",
    113: "CIN",
    114: "CLE",
    115: "COL",
    116: "DET",
    117: "HOU",
    118: "KC",
    119: "LAD",
    120: "WSH",
    121: "NYM",
    133: "OAK",
    134: "PIT",
    135: "SD",
    136: "SEA",
    137: "SF",
    138: "STL",
    139: "TB",
    140: "TEX",
    141: "TOR",
    142: "MIN",
    143: "PHI",
    144: "ATL",
    145: "CWS",
    146: "MIA",
    147: "NYY",
    158: "MIL",
}


def _get_season_start(season: int) -> datetime.date:
    """Return the MLB regular season start date, defaulting to March 26 if unknown."""
    return _SEASON_STARTS.get(season, datetime.date(season, 3, 26))


def _my_team_key(settings: LeagueSettings) -> str:
    """Return the Yahoo team key for my team.

    Prefers config setting (my_team_key), falls back to constructing from
    YAHOO_TEAM_ID env var.
    """
    if settings.my_team_key:
        return settings.my_team_key
    game_key = os.environ.get("YAHOO_GAME_KEY", "469")
    team_id = os.environ.get("YAHOO_TEAM_ID", "1")
    return f"{game_key}.l.{settings.league_id}.t.{team_id}"


# ── Step functions ─────────────────────────────────────────────────────────────


def _step_load_yahoo(
    conn: duckdb.DuckDBPyConnection,
    week: int,
) -> tuple[dict[str, int], pd.DataFrame]:
    """Load Yahoo Fantasy data into the database.

    Loads my roster, all rosters, transactions, player details, and free agents.
    Stages free agents in fact_waiver_scores with placeholder scores.

    Args:
        conn: Open DuckDB connection.
        week: Current fantasy week number.

    Returns:
        Tuple of (row_counts_per_table, free_agents_df).
        free_agents_df has player stats columns needed by rank_free_agents().
    """
    yahoo = YahooClient.from_env()
    row_counts: dict[str, int] = {}

    # My roster
    my_roster_df = yahoo.get_my_roster(week)
    row_counts[FACT_ROSTERS] = load_rosters(conn, my_roster_df)

    # All rosters (for matchup opponent tracking)
    all_rosters_df = yahoo.get_all_rosters(week)
    if not all_rosters_df.empty:
        row_counts[FACT_ROSTERS] = row_counts.get(FACT_ROSTERS, 0) + load_rosters(
            conn, all_rosters_df
        )

    # Transactions (last 7 days)
    txn_df = yahoo.get_transactions(days=7)
    row_counts[FACT_TRANSACTIONS] = load_transactions(conn, txn_df)

    # Matchup data (for opponent comparison)
    try:
        matchup_df = yahoo.get_current_matchup()
        row_counts[FACT_MATCHUPS] = load_matchups(conn, matchup_df)
    except Exception as exc:
        logger.warning("Matchup fetch failed (non-fatal): %s", exc)

    # Player details for rostered players
    player_ids: list[str] = (
        my_roster_df["player_id"].tolist() if not my_roster_df.empty else []
    )
    if not all_rosters_df.empty and "player_id" in all_rosters_df.columns:
        all_ids = all_rosters_df["player_id"].tolist()
        player_ids = list(set(player_ids) | set(all_ids))

    if player_ids:
        players_df = yahoo.get_player_details(player_ids)
        row_counts[DIM_PLAYERS] = load_players(conn, players_df)

    # Free agents — fetch with stats for waiver scoring.
    # Wrapped in try/except so a failure here does not abort the entire Yahoo step
    # (roster and player data above is more critical).
    try:
        fa_df = yahoo.get_free_agents(count=100)
        row_counts[FACT_WAIVER_SCORES] = stage_free_agents(conn, fa_df)

        # Load FA player details
        if not fa_df.empty and "player_id" in fa_df.columns:
            fa_ids: list[str] = fa_df["player_id"].tolist()
            fa_players_df = yahoo.get_player_details(fa_ids)
            existing = row_counts.get(DIM_PLAYERS, 0)
            row_counts[DIM_PLAYERS] = existing + load_players(conn, fa_players_df)
    except Exception as exc:
        logger.warning("Free agent fetch failed (non-fatal): %s", exc)

    logger.info("Yahoo load complete. Row counts: %s", row_counts)
    return row_counts, fa_df


def _step_load_mlb_stats(
    conn: duckdb.DuckDBPyConnection,
    today: datetime.date,
    week: int,
    season: int,
) -> tuple[dict[str, int], pd.DataFrame]:
    """Load MLB daily stats and build today's game schedule.

    Fetches yesterday's batter and pitcher stats (yesterday is the most recently
    completed game day) and today's schedule for lineup optimization.

    Args:
        conn: Open DuckDB connection.
        today: Today's date.
        week: Current fantasy week number.
        season: MLB season year.

    Returns:
        Tuple of (row_counts, schedule_player_df).
        schedule_player_df has column player_id (Yahoo format) for players
        whose teams play today.
    """
    row_counts: dict[str, int] = {}
    yesterday = today - datetime.timedelta(days=1)

    # Query rostered players from DB for stat fetching
    try:
        players_result = conn.execute(
            f"SELECT player_id, mlb_id, team FROM {DIM_PLAYERS} WHERE mlb_id IS NOT NULL"
        ).fetchdf()
    except duckdb.Error:
        players_result = pd.DataFrame(columns=["player_id", "mlb_id", "team"])

    # Fetch yesterday's batter stats (mlb_id → stat row)
    try:
        batter_stats = mlb_client.get_batter_stats(
            start_date=yesterday,
            end_date=yesterday,
        )
        if not batter_stats.empty and not players_result.empty:
            # Join to get Yahoo player_id via mlb_id
            batter_with_id = batter_stats.merge(
                players_result[["player_id", "mlb_id"]],
                on="mlb_id",
                how="inner",
            )
            if not batter_with_id.empty:
                batter_with_id["stat_date"] = yesterday
                row_counts["batter_daily"] = load_daily_stats(
                    conn, batter_with_id, yesterday
                )
    except Exception as exc:
        logger.warning("Batter stats fetch failed: %s", exc)

    # Fetch yesterday's pitcher stats
    try:
        pitcher_stats = mlb_client.get_pitcher_stats(
            start_date=yesterday,
            end_date=yesterday,
        )
        if not pitcher_stats.empty and not players_result.empty:
            pitcher_with_id = pitcher_stats.merge(
                players_result[["player_id", "mlb_id"]],
                on="mlb_id",
                how="inner",
            )
            if not pitcher_with_id.empty:
                pitcher_with_id["stat_date"] = yesterday
                existing = row_counts.get("batter_daily", 0)
                row_counts["batter_daily"] = existing + load_daily_stats(
                    conn, pitcher_with_id, yesterday
                )
    except Exception as exc:
        logger.warning("Pitcher stats fetch failed: %s", exc)

    # Build today's schedule as player-level DataFrame
    schedule_player_df = _build_player_schedule(today, players_result)

    logger.info("MLB stats load complete. Row counts: %s", row_counts)
    return row_counts, schedule_player_df


def _build_player_schedule(
    today: datetime.date,
    players_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a player-level schedule DataFrame for today's games.

    Fetches team-level schedule from MLB API, maps team IDs to abbreviations,
    then maps to Yahoo player_ids via dim_players.

    Args:
        today: Date to build schedule for.
        players_df: DataFrame with player_id and team columns from dim_players.

    Returns:
        DataFrame with column: player_id (Yahoo format) for players with games today.
    """
    try:
        team_schedule = mlb_client.get_daily_game_schedule(today)
    except Exception as exc:
        logger.warning("Schedule fetch failed: %s", exc)
        return pd.DataFrame(columns=["player_id"])

    if team_schedule.empty or "mlb_id" not in team_schedule.columns:
        return pd.DataFrame(columns=["player_id"])

    # Map team mlb_ids to abbreviations
    team_ids_today = set(team_schedule["mlb_id"].astype(int).tolist())
    team_abbrs_today = {
        _MLB_TEAM_ID_TO_ABBR[tid]
        for tid in team_ids_today
        if tid in _MLB_TEAM_ID_TO_ABBR
    }

    if players_df.empty or "team" not in players_df.columns:
        return pd.DataFrame(columns=["player_id"])

    # Filter players whose team plays today
    players_with_games = players_df[players_df["team"].isin(team_abbrs_today)][
        ["player_id"]
    ].copy()

    logger.info(
        "Schedule: %d teams play today → %d players with games",
        len(team_abbrs_today),
        len(players_with_games),
    )
    return players_with_games.reset_index(drop=True)


def _step_refresh_projections(
    conn: duckdb.DuckDBPyConnection,
    today: datetime.date,
    week: int,
    season: int,
) -> dict[str, int]:
    """Refresh pace-based projections from MLB Stats API season stats.

    Fetches season-to-date stats for rostered players and converts them
    to per-game rates for weekly projection. Gracefully skips on failure.

    Args:
        conn: Open DuckDB connection.
        today: Today's date (used as projection_date).
        week: Current fantasy week number.
        season: MLB season year.

    Returns:
        Row counts per table.
    """
    row_counts: dict[str, int] = {}

    try:
        # Get rostered players with mlb_ids
        try:
            players = conn.execute(
                f"SELECT player_id, mlb_id FROM {DIM_PLAYERS} WHERE mlb_id IS NOT NULL"
            ).fetchdf()
        except duckdb.Error:
            players = pd.DataFrame(columns=["player_id", "mlb_id"])

        if players.empty:
            logger.info("No players with mlb_id — skipping projections.")
            return row_counts

        mlb_ids: list[int] = players["mlb_id"].dropna().astype(int).tolist()
        proj_df = mlb_client.get_season_stats_for_projections(
            mlb_ids=mlb_ids, season=season
        )

        if proj_df.empty:
            logger.info("Pace-based projections empty — skipping.")
            return row_counts

        # Add required metadata columns
        proj_df = proj_df.copy()
        proj_df["projection_date"] = today
        proj_df["target_week"] = week

        # Join to get Yahoo player_id via mlb_id
        proj_with_id = proj_df.merge(
            players[["player_id", "mlb_id"]],
            on="mlb_id",
            how="inner",
        )

        if not proj_with_id.empty:
            row_counts[FACT_PROJECTIONS] = load_projections(conn, proj_with_id)

    except Exception as exc:
        logger.warning("Projection refresh failed (non-fatal): %s", exc)

    return row_counts


def _step_run_analysis(
    conn: duckdb.DuckDBPyConnection,
    today: datetime.date,
    week: int,
    season: int,
    settings: LeagueSettings,
    fa_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
) -> dict[str, object]:
    """Run matchup analysis, waiver ranking, and lineup optimization.

    Reads accumulated stats and projections from DB, runs all analysis
    functions, and returns the complete daily report dict.

    Args:
        conn: Open DuckDB connection.
        today: Today's date.
        week: Current fantasy week number.
        season: MLB season year.
        settings: League settings.
        fa_df: Free agents DataFrame (from Yahoo step, has stat columns).
        schedule_df: Player-level schedule DataFrame with player_id column.

    Returns:
        JSON-serializable daily report dict (output of build_daily_report()).
    """
    team_key = _my_team_key(settings)

    # ── 1. Get my roster ──────────────────────────────────────────────────────
    my_roster_df = _query_my_roster(conn, team_key, today, week, season)

    # ── 2. Get opponent team stats (from matchup) ─────────────────────────────
    opp_team_key = _query_opponent_team_key(conn, team_key, week, season)
    opp_roster_df = (
        _query_my_roster(conn, opp_team_key, today, week, season)
        if opp_team_key
        else pd.DataFrame(columns=["player_id"])
    )

    # ── 3. Get week-to-date stats ─────────────────────────────────────────────
    week_start = _get_week_start(today)
    my_stats_df = _query_week_stats(conn, my_roster_df, week_start, today)
    opp_stats_df = _query_week_stats(conn, opp_roster_df, week_start, today)

    # ── 4. Get projections for remaining games ────────────────────────────────
    my_proj_df = _query_projections(conn, my_roster_df, today, week)
    opp_proj_df = _query_projections(conn, opp_roster_df, today, week)

    # ── 5. Project week totals and score categories ───────────────────────────
    my_totals = project_week_totals(my_stats_df, my_proj_df)
    opp_totals = project_week_totals(opp_stats_df, opp_proj_df)

    # Aggregate to team-level (one row each)
    my_team_totals = _aggregate_to_team(my_totals)
    opp_team_totals = _aggregate_to_team(opp_totals)

    matchup_df = score_categories(
        my_team_totals, opp_team_totals, settings.category_win_direction
    )

    # ── 6. IP pace check ──────────────────────────────────────────────────────
    days_remaining = (week_start + datetime.timedelta(days=6) - today).days
    days_remaining = max(0, min(6, days_remaining))
    ip_pace = check_ip_pace(
        my_stats_df,
        days_remaining=days_remaining,
        min_ip=settings.min_ip_per_week,
    )

    # ── 7. Callup alerts ──────────────────────────────────────────────────────
    callup_alerts = _build_callup_alerts(conn, fa_df)

    # ── 8. Rank free agents ───────────────────────────────────────────────────
    callups_df = _query_callups(conn)
    ranked_fa_df = rank_free_agents(
        fa_df, my_roster_df, matchup_df, callups_df, settings
    )

    # Write real waiver scores back (overwrite placeholder 0-scores)
    _update_waiver_scores(conn, ranked_fa_df, today)

    # ── 9. Optimize lineup ────────────────────────────────────────────────────
    # Add accumulated_ip to my_roster_df for pitcher conservatism logic
    my_roster_with_ip = _enrich_roster_with_stats(my_roster_df, my_stats_df)
    lineup = optimize_daily_lineup(my_roster_with_ip, schedule_df, matchup_df, settings)

    # ── 10. Recommend adds ────────────────────────────────────────────────────
    acquisitions_used = _query_weekly_acquisitions(conn, team_key, today)
    adds = recommend_adds(ranked_fa_df, my_roster_df, acquisitions_used, settings)

    # ── 11. Build daily report ────────────────────────────────────────────────
    report = build_daily_report(
        lineup=lineup,
        adds=adds,
        matchup_df=matchup_df,
        ip_pace=ip_pace,
        callup_alerts=callup_alerts,
        report_date=today,
        week_number=week,
    )

    return report


def _step_write_daily_report(
    conn: duckdb.DuckDBPyConnection,
    report: dict[str, object],
    today: datetime.date,
    week: int,
    season: int,
    run_id: str,
) -> int:
    """Write the daily report JSON to fact_daily_reports.

    Args:
        conn: Open DuckDB connection.
        report: JSON-serializable daily report dict.
        today: Report date.
        week: Fantasy week number.
        season: MLB season year.
        run_id: UUID of the current pipeline run.

    Returns:
        Number of rows written (always 1 on success).
    """
    report_json = json.dumps(report)
    generated_at = datetime.datetime.now().isoformat()

    staging = pd.DataFrame(
        [
            {
                "report_date": today,
                "season": season,
                "week_number": week,
                "report_json": report_json,
                "generated_at": generated_at,
                "pipeline_run_id": run_id,
            }
        ]
    )
    conn.register("_report_staging", staging)
    conn.execute(f"""
        INSERT OR REPLACE INTO {FACT_DAILY_REPORTS}
            (report_date, season, week_number, report_json, generated_at, pipeline_run_id)
        SELECT report_date, season, week_number, report_json, generated_at, pipeline_run_id
        FROM _report_staging
    """)
    conn.unregister("_report_staging")
    logger.info("Daily report written for %s (week %d).", today, week)
    return 1


def _step_log_pipeline_run(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    status: str,
    rows_written: dict[str, int],
    errors: list[str],
    duration_seconds: float,
) -> None:
    """Write a pipeline run record to fact_pipeline_runs.

    Args:
        conn: Open DuckDB connection.
        run_id: UUID for this run.
        status: 'success' | 'partial' | 'failed'
        rows_written: Per-table row counts.
        errors: List of error messages (empty on success).
        duration_seconds: Total wall-clock time for the pipeline.
    """
    errors_str = "; ".join(errors) if errors else None
    staging = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "run_at": datetime.datetime.now().isoformat(),
                "status": status,
                "rows_written": json.dumps(rows_written),
                "errors": errors_str,
                "duration_seconds": round(duration_seconds, 2),
            }
        ]
    )
    conn.register("_run_staging", staging)
    conn.execute(f"""
        INSERT OR REPLACE INTO {FACT_PIPELINE_RUNS}
            (run_id, run_at, status, rows_written, errors, duration_seconds)
        SELECT run_id, run_at, status, rows_written, errors, duration_seconds
        FROM _run_staging
    """)
    conn.unregister("_run_staging")
    logger.info(
        "Pipeline run %s logged — status=%s, duration=%.1fs",
        run_id,
        status,
        duration_seconds,
    )


# ── Query helpers ──────────────────────────────────────────────────────────────


def _query_my_roster(
    conn: duckdb.DuckDBPyConnection,
    team_key: str,
    today: datetime.date,
    week: int,
    season: int,
) -> pd.DataFrame:
    """Query my current roster from fact_rosters + dim_players.

    Returns DataFrame with columns needed by the analysis functions:
    player_id, slot, eligible_positions, full_name, team, accumulated_ip.
    """
    try:
        df: pd.DataFrame = conn.execute(
            f"""
            SELECT
                r.player_id,
                r.roster_slot AS slot,
                p.positions AS eligible_positions,
                p.full_name,
                p.team,
                COALESCE(w.overall_score, 0.0) AS overall_score,
                0 AS games_remaining
            FROM {FACT_ROSTERS} r
            LEFT JOIN {DIM_PLAYERS} p ON r.player_id = p.player_id
            LEFT JOIN {FACT_WAIVER_SCORES} w
                ON r.player_id = w.player_id AND w.score_date = ?
            WHERE r.team_id = ?
              AND r.snapshot_date = ?
        """,
            [today, team_key, today],
        ).fetchdf()
    except duckdb.Error as exc:
        logger.warning("Roster query failed for %s: %s", team_key, exc)
        df = pd.DataFrame(
            columns=[
                "player_id",
                "slot",
                "eligible_positions",
                "full_name",
                "team",
                "overall_score",
                "games_remaining",
            ]
        )
    return df


def _query_opponent_team_key(
    conn: duckdb.DuckDBPyConnection,
    my_team_key: str,
    week: int,
    season: int,
) -> str | None:
    """Find the opposing team key from fact_matchups for the current week."""
    try:
        result = conn.execute(
            f"""
            SELECT
                CASE
                    WHEN team_id_home = ? THEN team_id_away
                    ELSE team_id_home
                END AS opponent_key
            FROM {FACT_MATCHUPS}
            WHERE (team_id_home = ? OR team_id_away = ?)
              AND week_number = ?
              AND season = ?
            LIMIT 1
        """,
            [my_team_key, my_team_key, my_team_key, week, season],
        ).fetchone()
        if result:
            return str(result[0])
    except duckdb.Error:
        pass
    return None


def _get_week_start(today: datetime.date) -> datetime.date:
    """Return the Monday of the current week."""
    return today - datetime.timedelta(days=today.weekday())


def _query_week_stats(
    conn: duckdb.DuckDBPyConnection,
    roster_df: pd.DataFrame,
    week_start: datetime.date,
    today: datetime.date,
) -> pd.DataFrame:
    """Query week-to-date daily stats for a roster's players.

    Args:
        conn: Open DuckDB connection.
        roster_df: Roster DataFrame with player_id column.
        week_start: Monday of the current fantasy week.
        today: Today's date (stats through yesterday).

    Returns:
        DataFrame of accumulated stats (sum over week), one row per player.
        Empty DataFrame if roster is empty or query fails.
    """
    _empty_stats = pd.DataFrame(
        columns=[
            "player_id",
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
        ]
    )
    if roster_df.empty or "player_id" not in roster_df.columns:
        return _empty_stats

    player_ids: list[str] = roster_df["player_id"].tolist()
    placeholders = ", ".join(["?" for _ in player_ids])

    try:
        df: pd.DataFrame = conn.execute(
            f"""
            SELECT
                player_id,
                SUM(COALESCE(ab, 0))            AS ab,
                SUM(COALESCE(h, 0))             AS h,
                SUM(COALESCE(hr, 0))            AS hr,
                SUM(COALESCE(sb, 0))            AS sb,
                SUM(COALESCE(bb, 0))            AS bb,
                SUM(COALESCE(hbp, 0))           AS hbp,
                SUM(COALESCE(sf, 0))            AS sf,
                SUM(COALESCE(tb, 0))            AS tb,
                SUM(COALESCE(errors, 0))        AS errors,
                SUM(COALESCE(chances, 0))       AS chances,
                SUM(COALESCE(ip, 0))            AS ip,
                SUM(COALESCE(w, 0))             AS w,
                SUM(COALESCE(k, 0))             AS k,
                SUM(COALESCE(walks_allowed, 0)) AS walks_allowed,
                SUM(COALESCE(hits_allowed, 0))  AS hits_allowed,
                SUM(COALESCE(sv, 0))            AS sv,
                SUM(COALESCE(holds, 0))         AS holds
            FROM fact_player_stats_daily
            WHERE player_id IN ({placeholders})
              AND stat_date >= ?
              AND stat_date < ?
            GROUP BY player_id
        """,
            player_ids + [week_start, today],
        ).fetchdf()
    except duckdb.Error as exc:
        logger.warning("Week stats query failed: %s", exc)
        df = _empty_stats

    return df


def _query_projections(
    conn: duckdb.DuckDBPyConnection,
    roster_df: pd.DataFrame,
    today: datetime.date,
    week: int,
) -> pd.DataFrame:
    """Query latest projections for a roster's players.

    Args:
        conn: Open DuckDB connection.
        roster_df: Roster DataFrame with player_id column.
        today: Today's date (matches projection_date).
        week: Target week number.

    Returns:
        DataFrame of projections, one row per player. Empty if none exist.
    """
    if roster_df.empty or "player_id" not in roster_df.columns:
        return pd.DataFrame(columns=["player_id"])

    player_ids: list[str] = roster_df["player_id"].tolist()
    placeholders = ", ".join(["?" for _ in player_ids])

    try:
        df: pd.DataFrame = conn.execute(
            f"""
            SELECT *
            FROM {FACT_PROJECTIONS}
            WHERE player_id IN ({placeholders})
              AND target_week = ?
            ORDER BY projection_date DESC
        """,
            player_ids + [week],
        ).fetchdf()
    except duckdb.Error as exc:
        logger.warning("Projections query failed: %s", exc)
        df = pd.DataFrame(columns=["player_id"])

    return df


def _aggregate_to_team(player_totals: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-player projected totals to a single team-level row.

    Counting stats are summed. Rate stats (avg, ops, fpct, whip, k_bb, sv_h)
    are recomputed from aggregated components to avoid averaging rates.

    Args:
        player_totals: Output of project_week_totals(), one row per player.

    Returns:
        One-row DataFrame with team totals for all 12 scoring categories.
    """
    if player_totals.empty:
        # Return a zero-row team total
        return pd.DataFrame(
            [
                {
                    "h": 0,
                    "hr": 0,
                    "sb": 0,
                    "bb": 0,
                    "avg": 0.0,
                    "ops": 0.0,
                    "fpct": 0.0,
                    "w": 0,
                    "k": 0,
                    "whip": 0.0,
                    "k_bb": 0.0,
                    "sv_h": 0,
                    "ab": 0,
                    "hbp": 0,
                    "sf": 0,
                    "tb": 0,
                    "errors": 0,
                    "chances": 0,
                    "ip": 0.0,
                    "walks_allowed": 0,
                    "hits_allowed": 0,
                    "sv": 0,
                    "holds": 0,
                }
            ]
        )

    counting = [
        "h",
        "hr",
        "sb",
        "bb",
        "hbp",
        "sf",
        "tb",
        "ab",
        "errors",
        "chances",
        "ip",
        "w",
        "k",
        "walks_allowed",
        "hits_allowed",
        "sv",
        "holds",
    ]

    agg: dict[str, Any] = {}
    for col in counting:
        if col in player_totals.columns:
            agg[col] = float(player_totals[col].fillna(0).sum())
        else:
            agg[col] = 0.0

    # Recompute rate stats from aggregated components
    ab = agg.get("ab", 0.0)
    h = agg.get("h", 0.0)
    bb = agg.get("bb", 0.0)
    hbp = agg.get("hbp", 0.0)
    sf = agg.get("sf", 0.0)
    tb = agg.get("tb", 0.0)
    chances = agg.get("chances", 0.0)
    errors = agg.get("errors", 0.0)
    ip = agg.get("ip", 0.0)
    walks_allowed = agg.get("walks_allowed", 0.0)
    hits_allowed = agg.get("hits_allowed", 0.0)
    k = agg.get("k", 0.0)
    sv = agg.get("sv", 0.0)
    holds = agg.get("holds", 0.0)

    agg["avg"] = h / ab if ab > 0 else 0.0
    obp_num = h + bb + hbp
    obp_denom = ab + bb + hbp + sf
    obp = obp_num / obp_denom if obp_denom > 0 else 0.0
    slg = tb / ab if ab > 0 else 0.0
    agg["ops"] = obp + slg
    agg["fpct"] = (chances - errors) / chances if chances > 0 else 0.0
    agg["whip"] = (walks_allowed + hits_allowed) / ip if ip > 0 else 0.0
    agg["k_bb"] = k / walks_allowed if walks_allowed > 0 else 0.0
    agg["sv_h"] = sv + holds

    return pd.DataFrame([agg])


def _build_callup_alerts(
    conn: duckdb.DuckDBPyConnection,
    fa_df: pd.DataFrame,
) -> list[dict[str, object]]:
    """Build callup alert dicts for recently called up free agents.

    Args:
        conn: Open DuckDB connection.
        fa_df: Free agents DataFrame; used to enrich with player names.

    Returns:
        List of callup alert dicts compatible with build_daily_report().
    """
    try:
        callups = mlb_client.get_recent_callups(days=7)
    except Exception as exc:
        logger.warning("Callup fetch failed: %s", exc)
        return []

    if callups.empty:
        return []

    # Enrich with player names from DB (include mlb_id for callup matching)
    try:
        player_names: pd.DataFrame = conn.execute(
            f"SELECT player_id, mlb_id, full_name, team FROM {DIM_PLAYERS}"
        ).fetchdf()
    except duckdb.Error:
        player_names = pd.DataFrame(
            columns=["player_id", "mlb_id", "full_name", "team"]
        )

    # fa_df may have player_id and full_name too — use as fallback
    if (
        not fa_df.empty
        and "player_id" in fa_df.columns
        and "full_name" in fa_df.columns
    ):
        fa_names = fa_df[["player_id", "full_name"]].rename(
            columns={"full_name": "fa_name"}
        )
    else:
        fa_names = pd.DataFrame(columns=["player_id", "fa_name"])

    # Build mlb_id → player_id lookup from dim_players
    mlb_to_yahoo: dict[int, str] = {}
    if not player_names.empty and "mlb_id" in player_names.columns:
        for _, prow in player_names.iterrows():
            mid = prow.get("mlb_id")
            if mid is not None and not (isinstance(mid, float) and pd.isna(mid)):
                mlb_to_yahoo[int(mid)] = str(prow["player_id"])

    alerts: list[dict[str, object]] = []
    for _, row in callups.iterrows():
        # Callups use mlb_id, not player_id — map through dim_players
        mlb_id = row.get("mlb_id")
        pid = mlb_to_yahoo.get(int(mlb_id), "") if mlb_id is not None else ""
        name_match = (
            player_names[player_names["player_id"] == pid] if pid else pd.DataFrame()
        )
        if not name_match.empty:
            name = str(name_match.iloc[0]["full_name"])
            team = str(name_match.iloc[0].get("team", ""))
        elif not fa_names.empty and pid:
            fa_match = fa_names[fa_names["player_id"] == pid]
            name = str(fa_match.iloc[0]["fa_name"]) if not fa_match.empty else ""
            team = ""
        else:
            name = ""
            team = ""

        # Fall back to the callup record's own name/team from MLB API
        if not name:
            name = str(row.get("full_name", ""))
        if not team:
            team = str(row.get("team", ""))

        # Compute days since callup from transaction_date
        txn_date = row.get("transaction_date")
        if txn_date is not None and hasattr(txn_date, "toordinal"):
            days_since = (datetime.date.today() - txn_date).days
        elif txn_date is not None:
            try:
                days_since = (
                    datetime.date.today()
                    - datetime.date.fromisoformat(str(txn_date)[:10])
                ).days
            except (ValueError, TypeError):
                days_since = 0
        else:
            days_since = int(row.get("days_since_callup", 0))

        alerts.append(
            {
                "player_id": pid,
                "player_name": name,
                "team": team,
                "from_level": str(row.get("from_level", "AAA")),
                "days_since_callup": days_since,
            }
        )

    return alerts


def _query_callups(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Query recent callups stored in fact_waiver_scores."""
    try:
        df: pd.DataFrame = conn.execute(f"""
            SELECT player_id, days_since_callup
            FROM {FACT_WAIVER_SCORES}
            WHERE is_callup = true
              AND score_date >= CURRENT_DATE - INTERVAL 7 DAYS
        """).fetchdf()
        return df
    except duckdb.Error:
        return pd.DataFrame(columns=["player_id", "days_since_callup"])


def _update_waiver_scores(
    conn: duckdb.DuckDBPyConnection,
    ranked_df: pd.DataFrame,
    today: datetime.date,
) -> None:
    """Overwrite placeholder waiver scores with real computed scores.

    Args:
        conn: Open DuckDB connection.
        ranked_df: Output of rank_free_agents() with overall_score etc.
        today: Score date to update.
    """
    if ranked_df.empty:
        return

    update_df = ranked_df.copy()
    update_df["score_date"] = today

    # Ensure all required columns exist
    for col in ["is_callup", "days_since_callup", "recommended_drop_id", "notes"]:
        if col not in update_df.columns:
            update_df[col] = None

    conn.register("_waiver_update", update_df)
    conn.execute(f"""
        INSERT OR REPLACE INTO {FACT_WAIVER_SCORES}
            (player_id, score_date, overall_score, category_scores,
             is_callup, days_since_callup, recommended_drop_id, notes)
        SELECT player_id, score_date, overall_score, category_scores,
               COALESCE(is_callup, false), days_since_callup,
               recommended_drop_id, notes
        FROM _waiver_update
    """)
    conn.unregister("_waiver_update")
    logger.info("Updated %d waiver scores for %s.", len(update_df), today)


def _enrich_roster_with_stats(
    roster_df: pd.DataFrame,
    stats_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add accumulated_ip to the roster DataFrame for pitcher conservatism."""
    if stats_df.empty or "ip" not in stats_df.columns:
        result = roster_df.copy()
        result["accumulated_ip"] = 0.0
        return result

    ip_by_player = stats_df[["player_id", "ip"]].rename(
        columns={"ip": "accumulated_ip"}
    )
    enriched = roster_df.merge(ip_by_player, on="player_id", how="left")
    enriched["accumulated_ip"] = enriched["accumulated_ip"].fillna(0.0)
    return enriched


def _query_weekly_acquisitions(
    conn: duckdb.DuckDBPyConnection,
    team_key: str,
    today: datetime.date,
) -> int:
    """Count add transactions made by my team this week.

    Args:
        conn: Open DuckDB connection.
        team_key: My Yahoo team key.
        today: Today's date (used to compute week start).

    Returns:
        Number of 'add' transactions made since Monday.
    """
    week_start = _get_week_start(today)
    try:
        result = conn.execute(
            f"""
            SELECT COUNT(*) AS n
            FROM {FACT_TRANSACTIONS}
            WHERE team_id = ?
              AND type = 'add'
              AND transaction_date >= ?
        """,
            [team_key, week_start],
        ).fetchone()
        return int(result[0]) if result else 0
    except duckdb.Error:
        return 0


# ── Public entry point ─────────────────────────────────────────────────────────


def run_daily_pipeline(
    conn: duckdb.DuckDBPyConnection,
    settings: LeagueSettings,
    run_date: datetime.date | None = None,
) -> dict[str, object]:
    """Run the full daily pipeline.

    Steps:
    1. Ensure schema exists
    2. Load Yahoo data (roster, transactions, free agents, players)
    3. Load MLB stats (daily stats + schedule)
    4. Refresh Steamer projections
    5. Run analysis (matchup, waiver, lineup)
    6. Write daily report
    7. Log pipeline run

    Each step is wrapped in try/except so one failure does not abort
    the entire pipeline. Status is 'success', 'partial', or 'failed'.

    Args:
        conn: Open DuckDB connection (MotherDuck or in-memory).
        settings: League settings from config/league_settings.yaml.
        run_date: Override today's date (for testing). Defaults to today.

    Returns:
        {
          "run_id": str,
          "status": "success" | "partial" | "failed",
          "rows_written": dict[str, int],
        }
    """
    run_id = str(uuid.uuid4())
    start_time = datetime.datetime.now()
    rows_written: dict[str, int] = {}
    errors: list[str] = []

    today = run_date if run_date is not None else datetime.date.today()
    season = today.year
    week = get_fantasy_week(today, _get_season_start(season))

    logger.info("Pipeline starting — run_id=%s, date=%s, week=%d", run_id, today, week)

    # Step 1: Ensure schema
    try:
        create_all_tables(conn)
    except Exception as exc:
        errors.append(f"schema: {exc}")
        logger.error("Schema creation failed: %s", exc)

    # Step 1b: Clean up stale stats that don't match any current player.
    # Old pipeline runs wrote stats with player_ids using wrong game key
    # prefixes (e.g. '422.p.*'). Remove stats rows whose player_id doesn't
    # exist in dim_players so they don't pollute queries.
    try:
        for table in [FACT_PLAYER_STATS_DAILY, FACT_PROJECTIONS]:
            conn.execute(
                f"DELETE FROM {table} WHERE player_id NOT IN "
                f"(SELECT player_id FROM {DIM_PLAYERS})"
            )
        logger.info("Cleaned orphaned stats rows.")
    except Exception as exc:
        logger.warning("Stale data cleanup failed (non-fatal): %s", exc)

    # Step 2: Load Yahoo data
    fa_df: pd.DataFrame = pd.DataFrame()
    try:
        yahoo_counts, fa_df = _step_load_yahoo(conn, week)
        rows_written.update(yahoo_counts)
    except Exception as exc:
        errors.append(f"yahoo: {exc}")
        logger.error("Yahoo load failed: %s", exc)

    # Step 2b: Crosswalk — update dim_players.mlb_id with real MLBAM IDs.
    # Yahoo stores its own internal player_id as mlb_id, which doesn't match
    # the MLB Stats API's MLBAM IDs. This step fetches active MLB players by
    # name and updates dim_players so the stats join in Step 3 works.
    try:
        active_players = mlb_client.get_active_mlb_players(season)
        if not active_players.empty:
            crosswalk_count = update_player_crosswalk(conn, active_players)
            rows_written["crosswalk"] = crosswalk_count
            logger.info("MLBAM crosswalk updated %d players.", crosswalk_count)
    except Exception as exc:
        errors.append(f"crosswalk: {exc}")
        logger.error("MLBAM crosswalk failed: %s", exc)

    # Step 3: Load MLB stats
    schedule_df: pd.DataFrame = pd.DataFrame(columns=["player_id"])
    try:
        mlb_counts, schedule_df = _step_load_mlb_stats(conn, today, week, season)
        rows_written.update(mlb_counts)
    except Exception as exc:
        errors.append(f"mlb: {exc}")
        logger.error("MLB stats load failed: %s", exc)

    # Step 4: Refresh projections
    try:
        proj_counts = _step_refresh_projections(conn, today, week, season)
        rows_written.update(proj_counts)
    except Exception as exc:
        errors.append(f"projections: {exc}")
        logger.error("Projection refresh failed: %s", exc)

    # Step 5: Run analysis
    report: dict[str, object] = {}
    try:
        report = _step_run_analysis(
            conn, today, week, season, settings, fa_df, schedule_df
        )
    except Exception as exc:
        errors.append(f"analysis: {exc}")
        logger.error("Analysis failed: %s", exc)

    # Step 6: Write daily report
    if report:
        try:
            rows_written[FACT_DAILY_REPORTS] = _step_write_daily_report(
                conn, report, today, week, season, run_id
            )
        except Exception as exc:
            errors.append(f"report_write: {exc}")
            logger.error("Report write failed: %s", exc)

    # Step 7: Log pipeline run
    duration = (datetime.datetime.now() - start_time).total_seconds()
    status = "success" if not errors else ("partial" if rows_written else "failed")

    try:
        _step_log_pipeline_run(conn, run_id, status, rows_written, errors, duration)
    except Exception as exc:
        logger.error("Pipeline run logging failed: %s", exc)

    logger.info(
        "Pipeline complete — run_id=%s, status=%s, duration=%.1fs",
        run_id,
        status,
        duration,
    )
    return {"run_id": run_id, "status": status, "rows_written": rows_written}


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    from src.config import load_league_settings
    from src.db.connection import managed_connection

    settings = load_league_settings()
    with managed_connection() as conn:
        result = run_daily_pipeline(conn, settings)
    print(f"Pipeline complete: {result}")
    sys.exit(0 if result["status"] != "failed" else 1)
