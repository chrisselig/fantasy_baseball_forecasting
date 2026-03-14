"""
schema.py

Single source of truth for all MotherDuck table definitions.

Rules:
- All column names are lowercase snake_case.
- All loaders and analysis modules import column name constants from here.
- Tables are created in dependency order (dimensions before facts).
- Rate stats in daily/weekly tables are stored as convenience columns only.
  Always recompute them from their raw components — never average rates directly.

OPS components stored per player per day:
  hbp   — hit by pitch (for OBP numerator)
  sf    — sacrifice fly (for OBP denominator)
  tb    — total bases (for SLG numerator)
  OBP   = (h + bb + hbp) / (ab + bb + hbp + sf)
  SLG   = tb / ab
  OPS   = OBP + SLG

WHIP is lowest-wins. All other categories are highest-wins.
"""

from __future__ import annotations

import logging

import duckdb

logger = logging.getLogger(__name__)

# ── Table names ───────────────────────────────────────────────────────────────

DIM_PLAYERS = "dim_players"
DIM_DATES = "dim_dates"
FACT_PLAYER_STATS_DAILY = "fact_player_stats_daily"
FACT_PLAYER_STATS_WEEKLY = "fact_player_stats_weekly"
FACT_ROSTERS = "fact_rosters"
FACT_TRANSACTIONS = "fact_transactions"
FACT_MATCHUPS = "fact_matchups"
FACT_WAIVER_SCORES = "fact_waiver_scores"
FACT_PROJECTIONS = "fact_projections"
FACT_DAILY_REPORTS = "fact_daily_reports"
FACT_PIPELINE_RUNS = "fact_pipeline_runs"

ALL_TABLES = [
    DIM_PLAYERS,
    DIM_DATES,
    FACT_PLAYER_STATS_DAILY,
    FACT_PLAYER_STATS_WEEKLY,
    FACT_ROSTERS,
    FACT_TRANSACTIONS,
    FACT_MATCHUPS,
    FACT_WAIVER_SCORES,
    FACT_PROJECTIONS,
    FACT_DAILY_REPORTS,
    FACT_PIPELINE_RUNS,
]

# ── DDL statements ────────────────────────────────────────────────────────────

_DDL_DIM_PLAYERS = f"""
CREATE TABLE IF NOT EXISTS {DIM_PLAYERS} (
    player_id   VARCHAR PRIMARY KEY,  -- Yahoo player key (e.g. '422.p.7578')
    mlb_id      INTEGER,              -- MLBAM ID — links to MLB Stats API + Statcast
    fg_id       VARCHAR,              -- FanGraphs ID — links to Steamer projections
    full_name   VARCHAR NOT NULL,
    team        VARCHAR,              -- MLB team abbreviation e.g. 'NYY'
    positions   VARCHAR[],            -- All eligible positions e.g. ['SS', '2B']
    bats        VARCHAR(1),           -- L / R / S
    throws      VARCHAR(1),           -- L / R
    status      VARCHAR,              -- Active | IL-10 | IL-60 | Minors | NA
    updated_at  TIMESTAMP
);
"""

_DDL_DIM_DATES = f"""
CREATE TABLE IF NOT EXISTS {DIM_DATES} (
    date            DATE PRIMARY KEY,
    season          INTEGER NOT NULL,
    week_number     INTEGER NOT NULL,  -- Fantasy week number (1-25)
    is_playoff_week BOOLEAN NOT NULL DEFAULT false,
    day_of_week     VARCHAR NOT NULL   -- Monday, Tuesday, ...
);
"""

_DDL_FACT_PLAYER_STATS_DAILY = f"""
CREATE TABLE IF NOT EXISTS {FACT_PLAYER_STATS_DAILY} (
    player_id       VARCHAR NOT NULL,
    stat_date       DATE NOT NULL,
    -- ── Batter raw components ──────────────────────────────────────────────
    ab              INTEGER,          -- at-bats (AVG denominator component)
    h               INTEGER,          -- hits
    hr              INTEGER,          -- home runs
    sb              INTEGER,          -- stolen bases
    bb              INTEGER,          -- walks (batter)
    hbp             INTEGER,          -- hit by pitch (OBP numerator component)
    sf              INTEGER,          -- sacrifice fly (OBP denominator component)
    tb              INTEGER,          -- total bases (SLG numerator: 1B+2*2B+3*3B+4*HR)
    errors          INTEGER,          -- fielding errors (FPCT component)
    chances         INTEGER,          -- total fielding chances: PO+A+E (FPCT denominator)
    -- ── Pitcher raw components ────────────────────────────────────────────
    ip              DECIMAL(5, 1),    -- innings pitched
    w               INTEGER,          -- wins
    k               INTEGER,          -- strikeouts
    walks_allowed   INTEGER,          -- BB allowed (WHIP + K/BB components)
    hits_allowed    INTEGER,          -- H allowed (WHIP component)
    sv              INTEGER,          -- saves
    holds           INTEGER,          -- holds
    -- ── Computed convenience columns (derived from raw components above) ──
    -- Recompute from components when aggregating — never average these directly.
    avg             DECIMAL(5, 3),    -- h / ab
    ops             DECIMAL(5, 3),    -- OBP + SLG
    fpct            DECIMAL(5, 3),    -- (chances - errors) / chances
    whip            DECIMAL(5, 3),    -- (walks_allowed + hits_allowed) / ip  [LOWEST WINS]
    k_bb            DECIMAL(5, 2),    -- k / walks_allowed
    sv_h            INTEGER,          -- sv + holds
    PRIMARY KEY (player_id, stat_date)
);
"""

_DDL_FACT_PLAYER_STATS_WEEKLY = f"""
CREATE TABLE IF NOT EXISTS {FACT_PLAYER_STATS_WEEKLY} (
    player_id       VARCHAR NOT NULL,
    week_number     INTEGER NOT NULL,
    season          INTEGER NOT NULL,
    -- ── Batter raw components (aggregated from daily) ────────────────────
    ab              INTEGER,
    h               INTEGER,
    hr              INTEGER,
    sb              INTEGER,
    bb              INTEGER,
    hbp             INTEGER,
    sf              INTEGER,
    tb              INTEGER,
    errors          INTEGER,
    chances         INTEGER,
    -- ── Pitcher raw components (aggregated from daily) ───────────────────
    ip              DECIMAL(6, 1),
    w               INTEGER,
    k               INTEGER,
    walks_allowed   INTEGER,
    hits_allowed    INTEGER,
    sv              INTEGER,
    holds           INTEGER,
    -- ── Computed rate stats (derived from aggregated components above) ───
    avg             DECIMAL(5, 3),
    ops             DECIMAL(5, 3),
    fpct            DECIMAL(5, 3),
    whip            DECIMAL(5, 3),    -- [LOWEST WINS]
    k_bb            DECIMAL(5, 2),
    sv_h            INTEGER,
    PRIMARY KEY (player_id, week_number, season)
);
"""

_DDL_FACT_ROSTERS = f"""
CREATE TABLE IF NOT EXISTS {FACT_ROSTERS} (
    team_id          VARCHAR NOT NULL,
    player_id        VARCHAR NOT NULL,
    snapshot_date    DATE NOT NULL,
    roster_slot      VARCHAR NOT NULL,   -- C | 1B | SP | BN | IL | NA | ...
    acquisition_type VARCHAR,            -- draft | waiver | trade | fa
    PRIMARY KEY (team_id, player_id, snapshot_date)
);
"""

_DDL_FACT_TRANSACTIONS = f"""
CREATE TABLE IF NOT EXISTS {FACT_TRANSACTIONS} (
    transaction_id   VARCHAR PRIMARY KEY,
    league_id        INTEGER NOT NULL,
    transaction_date TIMESTAMP NOT NULL,
    type             VARCHAR NOT NULL,   -- add | drop | trade
    team_id          VARCHAR NOT NULL,
    player_id        VARCHAR NOT NULL,
    from_team_id     VARCHAR,            -- NULL for adds from free agency
    notes            VARCHAR
);
"""

_DDL_FACT_MATCHUPS = f"""
CREATE TABLE IF NOT EXISTS {FACT_MATCHUPS} (
    matchup_id          VARCHAR PRIMARY KEY,  -- '87941_2026_W01_T3vsT7'
    league_id           INTEGER NOT NULL,
    week_number         INTEGER NOT NULL,
    season              INTEGER NOT NULL,
    team_id_home        VARCHAR NOT NULL,
    team_id_away        VARCHAR NOT NULL,
    -- ── Category totals (populated progressively during the week) ────────
    -- Batter categories
    h_home              INTEGER,  h_away              INTEGER,
    hr_home             INTEGER,  hr_away             INTEGER,
    sb_home             INTEGER,  sb_away             INTEGER,
    bb_home             INTEGER,  bb_away             INTEGER,
    avg_home            DECIMAL(5, 3),  avg_away      DECIMAL(5, 3),
    ops_home            DECIMAL(5, 3),  ops_away      DECIMAL(5, 3),
    fpct_home           DECIMAL(5, 3),  fpct_away     DECIMAL(5, 3),
    -- Pitcher categories
    w_home              INTEGER,  w_away              INTEGER,
    k_home              INTEGER,  k_away              INTEGER,
    whip_home           DECIMAL(5, 3),  whip_away     DECIMAL(5, 3),  -- LOWEST WINS
    k_bb_home           DECIMAL(5, 2),  k_bb_away     DECIMAL(5, 2),
    sv_h_home           INTEGER,  sv_h_away           INTEGER,
    -- ── Outcome (populated at end of week) ───────────────────────────────
    categories_won_home INTEGER,
    categories_won_away INTEGER,
    result              VARCHAR   -- home_win | away_win | tie
);
"""

_DDL_FACT_WAIVER_SCORES = f"""
CREATE TABLE IF NOT EXISTS {FACT_WAIVER_SCORES} (
    player_id            VARCHAR NOT NULL,
    score_date           DATE NOT NULL,
    overall_score        DECIMAL(8, 4) NOT NULL,
    category_scores      JSON,         -- per-category contribution scores
    is_callup            BOOLEAN NOT NULL DEFAULT false,
    days_since_callup    INTEGER,
    recommended_drop_id  VARCHAR,      -- player_id of suggested roster drop
    notes                VARCHAR,
    PRIMARY KEY (player_id, score_date)
);
"""

_DDL_FACT_PROJECTIONS = f"""
CREATE TABLE IF NOT EXISTS {FACT_PROJECTIONS} (
    player_id           VARCHAR NOT NULL,
    projection_date     DATE NOT NULL,
    target_week         INTEGER NOT NULL,
    -- ── Projected counting stats for remaining days in week ──────────────
    proj_h              DECIMAL(6, 2),
    proj_hr             DECIMAL(6, 2),
    proj_sb             DECIMAL(6, 2),
    proj_bb             DECIMAL(6, 2),
    proj_ab             DECIMAL(6, 2),  -- needed for AVG = (h+proj_h)/(ab+proj_ab)
    proj_tb             DECIMAL(6, 2),  -- needed for SLG = (tb+proj_tb)/(ab+proj_ab)
    proj_ip             DECIMAL(6, 2),
    proj_w              DECIMAL(6, 2),
    proj_k              DECIMAL(6, 2),
    proj_walks_allowed  DECIMAL(6, 2),  -- needed for WHIP + K/BB from components
    proj_hits_allowed   DECIMAL(6, 2),  -- needed for WHIP from components
    proj_sv_h           DECIMAL(6, 2),
    -- ── Projected rate stats (from Steamer/ZiPS — use for weighting) ─────
    proj_avg            DECIMAL(5, 3),
    proj_ops            DECIMAL(5, 3),
    proj_fpct           DECIMAL(5, 3),
    proj_whip           DECIMAL(5, 3),  -- [LOWEST WINS]
    proj_k_bb           DECIMAL(5, 2),
    -- ── Metadata ─────────────────────────────────────────────────────────
    games_remaining     INTEGER,
    source              VARCHAR NOT NULL,  -- steamer | zips | trailing_30 | fallback
    PRIMARY KEY (player_id, projection_date, target_week)
);
"""

_DDL_FACT_DAILY_REPORTS = f"""
CREATE TABLE IF NOT EXISTS {FACT_DAILY_REPORTS} (
    report_date      DATE PRIMARY KEY,
    season           INTEGER NOT NULL,
    week_number      INTEGER NOT NULL,
    report_json      JSON NOT NULL,    -- serialized output of lineup_optimizer.build_daily_report()
    generated_at     TIMESTAMP NOT NULL,
    pipeline_run_id  VARCHAR NOT NULL  -- FK to fact_pipeline_runs.run_id
);
"""

_DDL_FACT_PIPELINE_RUNS = f"""
CREATE TABLE IF NOT EXISTS {FACT_PIPELINE_RUNS} (
    run_id           VARCHAR PRIMARY KEY,  -- UUID generated at pipeline start
    run_at           TIMESTAMP NOT NULL,
    status           VARCHAR NOT NULL,     -- success | partial | failed
    rows_written     JSON,                 -- per-table row counts {{table: count}}
    errors           VARCHAR,              -- error message if status != success
    duration_seconds DECIMAL(8, 2)
);
"""

# Creation order matters — dimensions before facts that reference them
_DDL_IN_ORDER = [
    _DDL_DIM_PLAYERS,
    _DDL_DIM_DATES,
    _DDL_FACT_PLAYER_STATS_DAILY,
    _DDL_FACT_PLAYER_STATS_WEEKLY,
    _DDL_FACT_ROSTERS,
    _DDL_FACT_TRANSACTIONS,
    _DDL_FACT_MATCHUPS,
    _DDL_FACT_WAIVER_SCORES,
    _DDL_FACT_PROJECTIONS,
    _DDL_FACT_PIPELINE_RUNS,  # must exist before fact_daily_reports references it
    _DDL_FACT_DAILY_REPORTS,
]


# ── Public API ────────────────────────────────────────────────────────────────


def create_all_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all application tables if they do not already exist.

    Safe to call on a database that already has tables — uses
    CREATE TABLE IF NOT EXISTS throughout.

    Args:
        conn: An open DuckDB connection (MotherDuck or in-memory).
    """
    for ddl in _DDL_IN_ORDER:
        conn.execute(ddl)
    logger.info("All %d tables created/verified.", len(ALL_TABLES))


def drop_all_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Drop all application tables.

    Intended for test teardown only. Drops in reverse creation order
    to respect logical dependencies.

    Args:
        conn: An open DuckDB connection.
    """
    for table in reversed(ALL_TABLES):
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    logger.info("All %d tables dropped.", len(ALL_TABLES))


def get_existing_tables(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Return names of tables that currently exist in the database.

    Args:
        conn: An open DuckDB connection.

    Returns:
        List of existing table names.
    """
    result = conn.execute("SHOW TABLES").fetchall()
    return [row[0] for row in result]
