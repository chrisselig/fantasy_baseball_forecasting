"""
loaders_yahoo.py

ETL functions that write Yahoo Fantasy Sports API data into MotherDuck tables.

Each loader:
- Validates that required columns are present in the incoming DataFrame.
- Uses INSERT OR REPLACE semantics for idempotent upserts.
- Returns the number of rows written.
- Logs the row count on success.

Table name constants are imported from src.db.schema to ensure the
single-source-of-truth column naming is never duplicated here.
"""

from __future__ import annotations

import logging

import duckdb
import pandas as pd

from src.db.schema import (
    DIM_PLAYERS,
    FACT_MATCHUPS,
    FACT_ROSTERS,
    FACT_TRANSACTIONS,
    FACT_WAIVER_SCORES,
)

logger = logging.getLogger(__name__)


# ── Column requirements ────────────────────────────────────────────────────────

_REQUIRED_ROSTER_COLS = {
    "team_id",
    "player_id",
    "snapshot_date",
    "roster_slot",
}

_REQUIRED_TRANSACTION_COLS = {
    "transaction_id",
    "league_id",
    "transaction_date",
    "type",
    "team_id",
    "player_id",
}

_REQUIRED_PLAYER_COLS = {
    "player_id",
    "full_name",
}

_REQUIRED_FREE_AGENT_COLS = {
    "player_id",
}


# ── Validation helper ─────────────────────────────────────────────────────────


def _validate_columns(df: pd.DataFrame, required: set[str], table: str) -> None:
    """Raise ValueError if any required column is missing from *df*.

    Args:
        df: The DataFrame to validate.
        required: Set of column names that must be present.
        table: Target table name (used in the error message).

    Raises:
        ValueError: If one or more required columns are absent.
    """
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns for {table}: "
            f"{', '.join(sorted(missing))}"
        )


# ── Loaders ───────────────────────────────────────────────────────────────────


def load_rosters(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """Upsert a roster snapshot into fact_rosters.

    The primary key is (team_id, player_id, snapshot_date).  Rows with the
    same key are replaced so that re-running the pipeline on the same day is
    safe.

    Args:
        conn: Open DuckDB connection.
        df: DataFrame with at minimum the columns:
            team_id, player_id, snapshot_date, roster_slot.
            Optional: acquisition_type.

    Returns:
        Number of rows written.

    Raises:
        ValueError: If any required column is missing.
    """
    _validate_columns(df, _REQUIRED_ROSTER_COLS, FACT_ROSTERS)

    if df.empty:
        logger.info("load_rosters: DataFrame is empty — nothing to write.")
        return 0

    # Ensure optional column exists
    if "acquisition_type" not in df.columns:
        df = df.copy()
        df["acquisition_type"] = None

    insert_df = df[
        ["team_id", "player_id", "snapshot_date", "roster_slot", "acquisition_type"]
    ].copy()

    conn.register("_roster_staging", insert_df)
    conn.execute(f"""
        INSERT OR REPLACE INTO {FACT_ROSTERS}
            (team_id, player_id, snapshot_date, roster_slot, acquisition_type)
        SELECT team_id, player_id, snapshot_date, roster_slot, acquisition_type
        FROM _roster_staging
    """)
    conn.unregister("_roster_staging")

    row_count = len(insert_df)
    logger.info("load_rosters: wrote %d rows to %s.", row_count, FACT_ROSTERS)
    return row_count


def load_transactions(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """Upsert transactions into fact_transactions.

    The primary key is transaction_id.  Re-inserting an existing transaction
    replaces the row (idempotent).

    Args:
        conn: Open DuckDB connection.
        df: DataFrame with at minimum the columns:
            transaction_id, league_id, transaction_date, type, team_id, player_id.
            Optional: from_team_id, notes.

    Returns:
        Number of rows written.

    Raises:
        ValueError: If any required column is missing.
    """
    _validate_columns(df, _REQUIRED_TRANSACTION_COLS, FACT_TRANSACTIONS)

    if df.empty:
        logger.info("load_transactions: DataFrame is empty — nothing to write.")
        return 0

    # Ensure optional columns exist
    insert_df = df.copy()
    if "from_team_id" not in insert_df.columns:
        insert_df["from_team_id"] = None
    if "notes" not in insert_df.columns:
        insert_df["notes"] = None

    insert_df = insert_df[
        [
            "transaction_id",
            "league_id",
            "transaction_date",
            "type",
            "team_id",
            "player_id",
            "from_team_id",
            "notes",
        ]
    ]

    conn.register("_txn_staging", insert_df)
    conn.execute(f"""
        INSERT OR REPLACE INTO {FACT_TRANSACTIONS}
            (transaction_id, league_id, transaction_date, type,
             team_id, player_id, from_team_id, notes)
        SELECT transaction_id, league_id, transaction_date, type,
               team_id, player_id, from_team_id, notes
        FROM _txn_staging
    """)
    conn.unregister("_txn_staging")

    row_count = len(insert_df)
    logger.info("load_transactions: wrote %d rows to %s.", row_count, FACT_TRANSACTIONS)
    return row_count


def load_players(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """Upsert player dimension records into dim_players.

    The primary key is player_id.  Re-inserting replaces the row so that
    nightly refreshes pick up team/status changes.

    Args:
        conn: Open DuckDB connection.
        df: DataFrame with at minimum the columns: player_id, full_name.
            Optional: mlb_id, fg_id, team, positions, bats, throws,
                      status, updated_at.

    Returns:
        Number of rows written.

    Raises:
        ValueError: If any required column is missing.
    """
    _validate_columns(df, _REQUIRED_PLAYER_COLS, DIM_PLAYERS)

    if df.empty:
        logger.info("load_players: DataFrame is empty — nothing to write.")
        return 0

    insert_df = df.copy()

    optional_cols = {
        "mlb_id": None,
        "fg_id": None,
        "team": None,
        "positions": None,
        "bats": None,
        "throws": None,
        "status": "Active",
        "updated_at": None,
    }
    for col, default in optional_cols.items():
        if col not in insert_df.columns:
            insert_df[col] = default

    insert_df = insert_df[
        [
            "player_id",
            "mlb_id",
            "fg_id",
            "full_name",
            "team",
            "positions",
            "bats",
            "throws",
            "status",
            "updated_at",
        ]
    ]

    conn.register("_player_staging", insert_df)
    conn.execute(f"""
        INSERT OR REPLACE INTO {DIM_PLAYERS}
            (player_id, mlb_id, fg_id, full_name, team, positions,
             bats, throws, status, updated_at)
        SELECT player_id, mlb_id, fg_id, full_name, team, positions,
               bats, throws, status, updated_at
        FROM _player_staging
    """)
    conn.unregister("_player_staging")

    row_count = len(insert_df)
    logger.info("load_players: wrote %d rows to %s.", row_count, DIM_PLAYERS)
    return row_count


def stage_free_agents(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """Stage free agent data in fact_waiver_scores for downstream scoring.

    Writes the player_id and score_date as a staging record with a
    placeholder overall_score of 0.  The waiver ranker (Agent C) will
    overwrite these records with real scores.

    Args:
        conn: Open DuckDB connection.
        df: DataFrame with at minimum: player_id.
            Optionally: score_date, overall_score, category_scores,
                        is_callup, days_since_callup, recommended_drop_id,
                        notes.

    Returns:
        Number of rows staged.

    Raises:
        ValueError: If required column player_id is missing.
    """
    _validate_columns(df, _REQUIRED_FREE_AGENT_COLS, FACT_WAIVER_SCORES)

    if df.empty:
        logger.info("stage_free_agents: DataFrame is empty — nothing to stage.")
        return 0

    insert_df = df.copy()
    today = __import__("datetime").date.today().isoformat()

    defaults: dict[str, object] = {
        "score_date": today,
        "overall_score": 0.0,
        "category_scores": None,
        "is_callup": False,
        "days_since_callup": None,
        "recommended_drop_id": None,
        "notes": None,
    }
    for col, default in defaults.items():
        if col not in insert_df.columns:
            insert_df[col] = default  # type: ignore[call-overload]

    insert_df = insert_df[
        [
            "player_id",
            "score_date",
            "overall_score",
            "category_scores",
            "is_callup",
            "days_since_callup",
            "recommended_drop_id",
            "notes",
        ]
    ]

    conn.register("_fa_staging", insert_df)
    conn.execute(f"""
        INSERT OR REPLACE INTO {FACT_WAIVER_SCORES}
            (player_id, score_date, overall_score, category_scores,
             is_callup, days_since_callup, recommended_drop_id, notes)
        SELECT player_id, score_date, overall_score, category_scores,
               is_callup, days_since_callup, recommended_drop_id, notes
        FROM _fa_staging
    """)
    conn.unregister("_fa_staging")

    row_count = len(insert_df)
    logger.info(
        "stage_free_agents: staged %d rows in %s.", row_count, FACT_WAIVER_SCORES
    )
    return row_count


def load_matchups(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
) -> int:
    """Upsert matchup data into fact_matchups.

    Args:
        conn: Open DuckDB connection.
        df: DataFrame from ``YahooClient.get_current_matchup()``.
            Must include ``matchup_id`` column.

    Returns:
        Number of rows upserted.
    """
    if df.empty:
        logger.info("load_matchups: empty DataFrame, skipping.")
        return 0

    if "matchup_id" not in df.columns:
        logger.warning("load_matchups: missing matchup_id column.")
        return 0

    conn.register("_matchup_staging", df)
    conn.execute(
        f"INSERT OR REPLACE INTO {FACT_MATCHUPS} SELECT * FROM _matchup_staging"
    )
    conn.unregister("_matchup_staging")

    row_count = len(df)
    logger.info("load_matchups: upserted %d rows.", row_count)
    return row_count
