# Development Plan
## Fantasy Baseball Intelligence App

---

## Overview

This plan covers the full build from scaffolding to production deployment on shinyapps.io.
The architecture is split into two phases: a **Foundation** phase that must be completed
sequentially (it defines the contracts everything else depends on), followed by a
**Parallel Development** phase where four independent workstreams can run simultaneously
using git worktrees and subagents.

---

## Architecture: GitHub Actions as the Data Pipeline

The Shiny app on shinyapps.io is **read-only**. It never calls Yahoo or MLB APIs directly.
All data collection and analysis runs in GitHub Actions on a daily cron schedule and writes
results to MotherDuck. The app simply reads the pre-built report from MotherDuck on load.

This solves three problems at once:
- shinyapps.io instances sleep when idle — APScheduler won't fire on a sleeping instance
- API credentials (Yahoo OAuth tokens) stay in GitHub Secrets, not on shinyapps.io
- App cold-start is fast because the daily report is already computed before you open it

```
GitHub Actions (cron: daily 9am MT)
       │
       ├── yahoo_client.py   → fact_rosters, fact_transactions, fact_matchups
       ├── mlb_client.py     → dim_players, fact_player_stats_daily, callups
       ├── projections.py    → fact_projections
       │
       ├── matchup_analyzer  → score_categories()
       ├── waiver_ranker     → rank_free_agents()
       ├── lineup_optimizer  → build_daily_report()
       │
       └── writes daily_report → MotherDuck
                                      │
                              shinyapps.io reads
                              MotherDuck (read-only)
                                      │
                               Shiny app renders
```

## Dependency Map

```
[Project Scaffold]
       │
       ▼
[MotherDuck Schema + Connection]  ←── defines DataFrame contracts for all loaders
       │
       ├──────────────────────────────────────────────┐
       ▼                                              ▼
[Yahoo API Client]                          [MLB/Stats Data Client]
[Yahoo Loaders → MotherDuck]                [MLB Loaders → MotherDuck]
       │                                              │
       └────────────────────┬─────────────────────────┘
                            ▼
              [Analysis Modules]           [Shiny App Skeleton]
              matchup_analyzer             (built with mock data,
              waiver_ranker                wired to real analysis
              lineup_optimizer             in integration phase)
                            │                    │
                            └──────────┬──────────┘
                                       ▼
                        [GitHub Actions Pipeline wiring]
                                       │
                                       ▼
                                  [Deployment]
```

---

## Phase 1 — Foundation
### Must be completed before parallel work begins

This phase defines the schema, connection patterns, and DataFrame contracts
that all other modules depend on. It is small and fast to build.

### 1.1 Project Scaffolding

Set up the Python project structure per `CLAUDE.md`.

**Tasks:**
- Create `pyproject.toml` with all dependencies and project metadata
- Create `requirements.txt` for shinyapps.io deployment
- Create `.venv` and install all dependencies
- Create all `src/` subdirectories with `__init__.py` files
- Create `tests/` directory structure mirroring `src/`
- Create `.github/workflows/daily_pipeline.yml` skeleton (cron trigger, secrets references)
- Create `.env.example` documenting all required environment variables
- Configure `ruff`, `mypy`, and `pytest` in `pyproject.toml`
- Verify `ruff format .`, `ruff check .`, `mypy .`, `pytest` all pass on empty project

**Deliverable:** A clean, runnable Python project with all tooling configured.

---

### 1.2 MotherDuck Connection (`src/db/connection.py`)

Single module responsible for all MotherDuck connectivity.

**Tasks:**
- Implement `get_connection() -> duckdb.DuckDBPyConnection`
  - Reads `MOTHERDUCK_TOKEN` from environment
  - Returns `duckdb.connect("md:fantasy_baseball")` in production
  - Returns `duckdb.connect(":memory:")` when `MOTHERDUCK_TOKEN` is not set (local dev / tests)
- Implement a context manager wrapper for safe connection handling
- Write tests using in-memory DuckDB — never real MotherDuck in tests

---

### 1.3 Schema Definition (`src/db/schema.py`)

All `CREATE TABLE IF NOT EXISTS` statements in one place.

**Tasks:**
- Implement `create_all_tables(conn)` that creates all 8 tables in dependency order:
  1. `dim_players`
  2. `dim_dates`
  3. `fact_player_stats_daily`
  4. `fact_player_stats_weekly`
  5. `fact_rosters`
  6. `fact_transactions`
  7. `fact_matchups`
  8. `fact_waiver_scores`
  9. `fact_projections`
- Implement `drop_all_tables(conn)` for test teardown
- Write tests: create all tables on in-memory DB, assert all exist
- **This file is the single source of truth for column names and types.**
  All loaders and analysis modules reference these column names.

---

### 1.4 Config Loader (`src/config.py`)

Typed access to `config/league_settings.yaml`.

**Tasks:**
- Implement `load_league_settings() -> LeagueSettings` (dataclass)
- Expose: `league_id`, `scoring_categories`, `category_win_direction`
  (dict mapping each category to `"highest"` or `"lowest"`),
  `roster_slots`, `min_ip_per_week`, `max_acquisitions_per_week`,
  `playoff_start_week`, `trade_end_date`
- This config object is passed into analysis modules — no module reads the YAML directly
- Write tests: load from real YAML, assert all fields parse correctly

---

## Phase 2 — Parallel Development
### Four independent workstreams, four git worktrees, four subagents

Once Phase 1 is merged to `main`, all four agents branch from that state and work
independently. Each agent has a clearly bounded scope and defined output contracts.
They communicate only through the schema (already defined) and the DataFrame
interfaces documented below.

---

### Worktree A — Yahoo API Client
**Branch:** `feature/yahoo-api-client`

This agent builds the complete Yahoo Fantasy Sports API integration.
It does not implement any analysis logic — it only fetches and normalizes data.

#### A.1 OAuth Authentication (`src/api/yahoo_client.py`)

Yahoo client ID and secret are already obtained. The initial OAuth flow must be run
locally (requires browser redirect) to generate the access and refresh tokens. These
tokens are then stored in GitHub Secrets and used by the Actions pipeline — they are
never needed on shinyapps.io.

- Implement `YahooClient` class initialized with `consumer_key`, `consumer_secret`,
  `access_token`, `refresh_token` from environment variables
- Implement `_refresh_token_if_needed()` — called before every request; Yahoo access
  tokens expire after 1 hour, refresh tokens last much longer
- Implement `get_connection() -> YahooClient` — reads all four values from env, raises
  a clear `EnvironmentError` if any are missing
- Implement `_get(endpoint: str) -> dict` — base authenticated GET with error handling
- Raise `YahooAuthError` on 401/403, `YahooAPIError` on other failures
- Log all request URLs at DEBUG, response status codes at INFO
- **First task before any other Yahoo work:** run the one-time local auth script to
  generate and verify tokens work against the real league (ID: 87941)

#### A.2 Data Fetch Methods

Each method returns a normalized `pd.DataFrame` matching the column names in `schema.py`.

| Method | Returns | MotherDuck target |
|---|---|---|
| `get_my_roster(week)` | Player roster with slots | `fact_rosters` |
| `get_all_rosters(week)` | All 10 teams' rosters | `fact_rosters` |
| `get_current_matchup()` | Both teams' week-to-date stats | `fact_matchups` (partial) |
| `get_free_agents(count=100)` | Available players + stats | staging for `fact_waiver_scores` |
| `get_transactions(days=7)` | Recent adds/drops/trades | `fact_transactions` |
| `get_player_details(player_ids)` | Name, team, positions, status | `dim_players` |
| `get_standings()` | Current league standings | reference only |

#### A.3 Yahoo Loaders (`src/db/loaders.py` — Yahoo section)

- `load_rosters(conn, df)` → upsert into `fact_rosters`
- `load_transactions(conn, df)` → upsert into `fact_transactions`
- `load_players(conn, df)` → upsert into `dim_players`
- `stage_free_agents(conn, df)` → stage for waiver scoring

#### A.4 Tests

- Mock all HTTP calls using `pytest-mock` / `responses` library
- Test token refresh logic (expired token triggers refresh before retry)
- Test each fetch method returns a DataFrame with the correct columns
- Test loaders on in-memory DuckDB
- Test error handling: 401, 429, malformed response

---

### Worktree B — MLB & Stats Data Client
**Branch:** `feature/mlb-data-client`

This agent builds all non-Yahoo data ingestion: live MLB stats, player info,
call-up/transaction tracking, and projection ingestion.

#### B.1 MLB Transactions Feed (`src/api/mlb_client.py`)

- Implement `get_recent_callups(days: int = 7) -> pd.DataFrame`
  - Source: MLB Stats API transactions endpoint (public, no auth)
  - Filter for `typeCode = "CU"` (call-up) transactions
  - Return columns: `mlb_id`, `full_name`, `team`, `transaction_date`, `from_level`
- Implement `get_player_info(mlb_id: int) -> dict`
  - Source: MLB Stats API people endpoint
  - Return: name, team, position, bats, throws, active status
- Implement `get_daily_game_schedule(date: datetime.date) -> pd.DataFrame`
  - Return: `mlb_id`, `game_date`, `opponent_team`, `home_away`, `probable_pitcher`
  - Used by lineup optimizer to know who plays today

#### B.2 Statcast / pybaseball Integration

- Implement `get_batter_stats(start_date, end_date) -> pd.DataFrame`
  - Source: `pybaseball.batting_stats()` (FanGraphs-backed, season-level)
  - For daily granularity: `pybaseball.statcast()` filtered by batter
  - Normalize to `fact_player_stats_daily` columns
  - Cache all pybaseball results in MotherDuck — pybaseball calls are slow and rate-limited
- Implement `get_pitcher_stats(start_date, end_date) -> pd.DataFrame`
  - Source: `pybaseball.pitching_stats()` for season stats
  - For daily: MLB Stats API `/people/{id}/stats?stats=gameLog` (official, reliable)
  - Normalize to `fact_player_stats_daily` columns (IP, W, K, BB, H, SV, HLD)
- Implement `get_minor_league_stats(mlb_id: int) -> pd.DataFrame`
  - Source: MLB Stats API `/people/{id}/stats?stats=season&sportId=11` (sportId 11=AAA,
    12=AA, 13=A+, 14=A) — this is the correct endpoint, NOT pybaseball
  - Used to evaluate call-up quality: AVG, HR, K%, BB% at most recent MiLB level

#### B.3 Projections Ingestion

**Note:** FanGraphs scraping via pybaseball is fragile and may violate ToS. Use the
following strategy in priority order:

1. **Primary:** `pybaseball.fg_batting_projections()` / `pybaseball.fg_pitching_projections()`
   - These pull Steamer projections from FanGraphs public pages
   - Cache immediately in MotherDuck on first successful pull for the season
   - If FanGraphs blocks the request, fall back to option 2
2. **Fallback:** Baseball Savant's publically available depth chart projections
   - `pybaseball.statcast_batter_exitvelo_barrels()` for Statcast-based estimates
3. **Last resort:** derive simple projections from trailing 30-day stats in MotherDuck

- Implement `get_steamer_projections(season: int) -> pd.DataFrame`
  - Attempt FanGraphs pull; on failure log warning and use cached or fallback
  - Normalize to `fact_projections` columns, set `source = "steamer"` or `"fallback"`
- Map FanGraphs player IDs to Yahoo player IDs via `dim_players.mlb_id`
- **Important:** FanGraphs uses their own integer IDs (fgid), MLB Stats API uses MLBAM IDs.
  `pybaseball` has a `playerid_lookup()` function that maps between them — use this to
  build the ID crosswalk during the first load.

#### B.4 MLB Loaders (`src/db/loaders.py` — MLB section)

- `load_daily_stats(conn, df, stat_date)` → upsert into `fact_player_stats_daily`
- `load_weekly_stats(conn, df, week)` → rebuild `fact_player_stats_weekly`
- `load_projections(conn, df)` → upsert into `fact_projections`
- `load_dim_dates(conn, season, start_date, end_date)` → populate `dim_dates`

#### B.5 Tests

- Mock `pybaseball` calls to return fixture DataFrames
- Mock MLB Stats API HTTP responses
- Test player ID mapping (FanGraphs → Yahoo via mlb_id)
- Test `load_daily_stats` accumulates correctly across multiple days
- Test call-up detection returns correct players for a mocked transaction window

---

### Worktree C — Analysis Modules
**Branch:** `feature/analysis-modules`

This agent builds all three analysis modules using **fixture DataFrames only** —
no real DB or API calls. Input/output contracts are defined by the schema from Phase 1.

All analysis functions must be pure: accept DataFrames, return DataFrames.
No DB connections. No API calls. Testable in complete isolation.

#### C.1 Matchup Analyzer (`src/analysis/matchup_analyzer.py`)

**Inputs:**
- `my_stats: pd.DataFrame` — my team's week-to-date stats (columns from `fact_player_stats_daily`)
- `opp_stats: pd.DataFrame` — opponent's week-to-date stats
- `my_projections: pd.DataFrame` — remaining-week projections for my rostered players
- `opp_projections: pd.DataFrame` — remaining-week projections for opponent
- `category_config: dict` — from `LeagueSettings.category_win_direction`

**Key functions:**
- `project_week_totals(stats_df, projections_df) -> pd.DataFrame`
  - Combine accumulated stats + projected remaining games
  - Handle rate stats carefully: AVG = (H + proj_H) / (AB + proj_AB), not a simple sum
  - Same logic for OPS, FPCT, WHIP, K/BB
- `score_categories(my_totals, opp_totals, config) -> pd.DataFrame`
  - Returns one row per category with: `category`, `my_value`, `opp_value`,
    `my_leads`, `margin_pct`, `win_prob`, `status`
  - `status` ∈ `["safe_win", "flippable_win", "toss_up", "flippable_loss", "safe_loss"]`
  - For WHIP: lower is better — invert the comparison
- `get_focus_categories(scored_df) -> list[str]`
  - Return categories where lineup decisions this week matter (flippable win/loss, toss-up)
- `check_ip_pace(my_stats_df, days_remaining) -> dict`
  - Return: `{"current_ip": float, "projected_ip": float, "min_ip": 21, "on_pace": bool}`

**Tests:** Provide realistic fixture DataFrames for a mid-week matchup scenario.
Test the WHIP inversion explicitly. Test rate stat projection math. Test IP pace logic.

---

#### C.2 Waiver Wire Ranker (`src/analysis/waiver_ranker.py`)

**Inputs:**
- `free_agents: pd.DataFrame` — available players with season stats and projections
- `my_roster: pd.DataFrame` — current roster with stats
- `matchup_analysis: pd.DataFrame` — output of `score_categories()`
- `callups: pd.DataFrame` — recent MLB call-ups (from mlb_client)
- `config: LeagueSettings`

**Key functions:**
- `score_free_agent(player_row, my_roster_df, matchup_df, config) -> dict`
  - Compute `category_scores`: for each category, how much does adding this player improve
    my projected total vs. replacing the weakest player at that position?
  - Weight scores by `matchup_df.status`: improvements in flippable categories score higher
  - Return: `player_id`, `overall_score`, `category_scores` (JSON), `recommended_drop_id`
- `rank_free_agents(free_agents_df, ...) -> pd.DataFrame`
  - Apply `score_free_agent` across all available players
  - Sort descending by `overall_score`
  - Add `is_callup: bool` and `days_since_callup: int` (join with callups DataFrame)
- `find_recommended_drop(player_id, my_roster_df, config) -> str`
  - Given a player to add, who is the best drop? Match by position eligibility.
  - Factor in remaining schedule: don't drop a player with 5 games left this week.

**Tests:** Test scoring correctly weights flippable categories. Test call-up flag assignment.
Test that recommended drops respect positional eligibility. Test WHIP direction.

---

#### C.3 Lineup Optimizer (`src/analysis/lineup_optimizer.py`)

**Inputs:**
- `my_roster: pd.DataFrame` — full roster including BN/IL/NA
- `today_schedule: pd.DataFrame` — who plays today (from mlb_client)
- `matchup_analysis: pd.DataFrame` — output of `score_categories()`
- `waiver_ranking: pd.DataFrame` — output of `rank_free_agents()`
- `config: LeagueSettings`

**Key functions:**
- `optimize_daily_lineup(roster_df, schedule_df, matchup_df, config) -> dict`
  - Fill each active slot (C, 1B, 2B, 3B, SS, OF×3, Util×2, SP×2, RP×2, P×2)
  - Only start players who have a game today
  - For Util slots: prefer players who help in flippable categories
  - For pitching: track accumulated IP; if above 19 IP, be conservative starting more pitchers
  - Return: `{"slot": "player_id"}` mapping
- `recommend_adds(waiver_df, my_roster_df, acquisitions_used, max_per_week) -> list[dict]`
  - Filter waiver ranking to only actionable adds (within weekly acquisition limit)
  - Return top N with: `add_player_id`, `drop_player_id`, `reason`, `categories_improved`
- `build_daily_report(lineup, adds, matchup_df, ip_pace) -> dict`
  - Aggregate all recommendations into a single structured report consumed by the Shiny app

**Tests:** Test that inactive players (no game today) are not started. Test Util slot
logic favors flippable categories. Test IP threshold conservatism. Test acquisition
limit enforcement.

---

### Worktree D — Shiny App
**Branch:** `feature/shiny-app`

This agent builds the complete Shiny for Python application using **mock/stub data**
for all data sources. The app structure, UI layout, and reactive logic are fully built
here; real data gets wired in during integration.

#### D.1 App Entry Point (`src/app/app.py`)

- Import `ui` and `server` from their respective modules
- Create the app: `app = App(app_ui, server)`
- Handle startup: attempt MotherDuck connection, log success/failure, set a
  `data_available` reactive flag the UI reads to show an empty state gracefully

#### D.2 UI Layout (`src/app/ui.py`)

Build the full four-tab layout:

**Tab 1: Dashboard**
```
┌─────────────────────────────────────────────────────────────┐
│  Today: [date]   Week [N]   vs. [Opponent]   IP: 14.2/21   │
├──────────────────────────┬──────────────────────────────────┤
│  TODAY'S LINEUP          │  MATCHUP SCOREBOARD              │
│  ┌──────┬─────────────┐  │  ┌──────┬──────┬──────┬───────┐ │
│  │ Slot │ Player      │  │  │ Cat  │ Mine │ Opp  │ Prob  │ │
│  │ C    │ ...         │  │  │ H    │ 42   │ 38   │  73%  │ │
│  │ 1B   │ ...         │  │  │ HR   │ 8    │ 9    │  41%  │ │
│  │ ...  │ ...         │  │  │ WHIP │ 1.18 │ 1.24 │  61%  │ │
│  └──────┴─────────────┘  │  │ ...  │ ...  │ ...  │  ...  │ │
│                           │  └──────┴──────┴──────┴───────┘ │
├──────────────────────────┴──────────────────────────────────┤
│  RECOMMENDED ADDS                                           │
│  ┌────────────┬──────────┬──────┬──────────────────────┐   │
│  │ Add        │ Drop     │Score │ Why                  │   │
│  │ [Player A] │[Player X]│ 8.4  │ +K, +SV+H (flippable)│  │
│  │ [Player B] │[Player Y]│ 6.1  │ +HR, +SB             │   │
│  └────────────┴──────────┴──────┴──────────────────────┘   │
│  🚨 CALL-UP ALERTS: [Player C] promoted from AAA (2d ago)  │
└─────────────────────────────────────────────────────────────┘
```

**Tab 2: Matchup** — Full category breakdown, projected end-of-week totals,
  win probability bars, focus category highlights

**Tab 3: Waiver Wire** — Full ranked free agent table with filters for position,
  category impact, call-up flag; sortable columns

**Tab 4: Roster** — Current roster with season stats per category, strength
  indicator per category vs. league average

#### D.3 Server Logic (`src/app/server.py`)

Define all reactive components. Use mock data stubs initially:

```python
# Stub — replaced in integration phase
def _load_daily_report() -> dict:
    """Load today's recommendations. Returns mock data if DB unavailable."""
    ...

@reactive.calc
def daily_report():
    return _load_daily_report()

@reactive.calc
def matchup_data():
    return _load_matchup_data()

@reactive.calc
def waiver_data():
    return _load_waiver_data()
```

- All expensive computations go in `@reactive.calc` (cached, not re-run on every render)
- Outputs only reference reactive calcs — never call analysis functions directly
- Include a manual refresh button that invalidates all reactive calcs

#### D.4 Stub Data (`src/app/stubs.py`)

Realistic mock DataFrames matching every schema contract. Used during development
and as fallback when DB is unavailable. Enables the UI to be fully built and
tested without a live data connection.

#### D.5 Tests

- Test that each output renders without error given stub data
- Test that the IP pace indicator shows correct color (green/yellow/red)
- Test that call-up alert section appears/disappears correctly
- Test manual refresh button invalidates reactive calcs

---

## Phase 3 — Integration
### Sequential, builds on all four worktrees merged to main

### 3.1 Wire Data Pipeline to Analysis

Replace stubs in `src/app/server.py` with real calls:

```
YahooClient.get_my_roster()     → fact_rosters
YahooClient.get_current_matchup() → fact_matchups
MLBClient.get_daily_stats()     → fact_player_stats_daily
MLBClient.get_projections()     → fact_projections
MLBClient.get_callups()         → used directly by waiver_ranker
          ↓
matchup_analyzer.project_week_totals()
matchup_analyzer.score_categories()
          ↓
waiver_ranker.rank_free_agents()
          ↓
lineup_optimizer.optimize_daily_lineup()
lineup_optimizer.recommend_adds()
lineup_optimizer.build_daily_report()
          ↓
Shiny server reactive calcs
```

### 3.2 GitHub Actions Daily Pipeline

Replace APScheduler entirely with a GitHub Actions cron workflow. shinyapps.io instances
sleep when idle — any in-process scheduler will miss its triggers. GitHub Actions runs
independently of the app on a reliable schedule.

Create `.github/workflows/daily_pipeline.yml`:

```yaml
name: Daily Fantasy Baseball Pipeline

on:
  schedule:
    - cron: '0 15 * * *'  # 9am MT (UTC-6) every day during season
  workflow_dispatch:        # allow manual trigger any time

jobs:
  run-pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: python -m src.pipeline.daily_run
        env:
          MOTHERDUCK_TOKEN: ${{ secrets.MOTHERDUCK_TOKEN }}
          YAHOO_CONSUMER_KEY: ${{ secrets.YAHOO_CONSUMER_KEY }}
          YAHOO_CONSUMER_SECRET: ${{ secrets.YAHOO_CONSUMER_SECRET }}
          YAHOO_ACCESS_TOKEN: ${{ secrets.YAHOO_ACCESS_TOKEN }}
          YAHOO_REFRESH_TOKEN: ${{ secrets.YAHOO_REFRESH_TOKEN }}
```

Create `src/pipeline/daily_run.py` — the pipeline entry point:
1. Fetch Yahoo roster, matchup, transactions → write to MotherDuck
2. Fetch MLB daily stats → write to MotherDuck
3. Fetch/refresh projections (weekly, not daily) → write to MotherDuck
4. Run matchup_analyzer, waiver_ranker, lineup_optimizer
5. Write `daily_report` output to MotherDuck `fact_daily_reports` table
6. Log run metadata (timestamp, rows written, errors) to MotherDuck

**GitHub Secrets required:**
- `MOTHERDUCK_TOKEN`
- `YAHOO_CONSUMER_KEY`, `YAHOO_CONSUMER_SECRET`
- `YAHOO_ACCESS_TOKEN`, `YAHOO_REFRESH_TOKEN`

The Shiny app on shinyapps.io only needs `MOTHERDUCK_TOKEN` — no Yahoo credentials there.

### 3.3 Player ID Mapping

The hardest integration problem. Yahoo player IDs, MLB IDs, and FanGraphs IDs
are all different systems. Build and maintain a mapping table:

- On first run: fetch all rostered + free agent players from Yahoo (get Yahoo IDs)
- Cross-reference with `pybaseball` player lookup tables to get MLB IDs
- Store mapping in `dim_players`: `player_id` (Yahoo), `mlb_id` (MLB/Statcast/FanGraphs)
- Log any unmatched players for manual review

### 3.4 End-to-End Test

Write `tests/integration/test_pipeline.py`:
- Uses a pre-seeded in-memory DuckDB with fixture data
- Runs the full pipeline: analysis → report generation → assert report structure is correct
- Does NOT call real Yahoo or MLB APIs

---

## Phase 4 — Deployment
### One-time setup + ongoing

### 4.1 Yahoo OAuth Token — One-Time Local Setup

This must be done before any pipeline work. Yahoo OAuth requires a browser redirect
on first auth, which can only happen locally.

```bash
# Run once locally to generate tokens
python scripts/yahoo_auth.py
```

`scripts/yahoo_auth.py` will:
1. Read `YAHOO_CONSUMER_KEY` and `YAHOO_CONSUMER_SECRET` from `.env`
2. Open a browser for Yahoo login + authorize
3. Write `access_token` and `refresh_token` to stdout (do not write to file)
4. Store all four values in GitHub Secrets manually

Tokens then live in GitHub Secrets and are passed to the Actions pipeline via env vars.
The pipeline refreshes the access token automatically on every run (1-hour expiry).
**shinyapps.io never needs Yahoo credentials.**

### 4.2 shinyapps.io Configuration

shinyapps.io only needs one secret — MotherDuck. The app is read-only.

- Set environment variable via shinyapps.io dashboard: `MOTHERDUCK_TOKEN` only
- Run `/deploy` command checklist before every production push
- Upgrade to Starter plan ($9/month) — the free tier's 25 active hours/month will not
  last a 6-month baseball season

### 4.3 Monitoring

- Log all data refresh events with timestamps to MotherDuck
- If Yahoo API fails, app shows last-known data with a "data as of [datetime]" banner
- If MotherDuck is unreachable, app shows stub data with a clear offline notice

---

## Worktree & Subagent Recommendation

### When to launch agents

Launch Phase 2 agents **only after Phase 1 is merged to main.** The schema and config
loader are the contracts everything else builds against — starting before they exist
creates merge conflicts and rework.

### Recommended: 4 parallel subagents

| Agent | Worktree Branch | Scope | Key outputs |
|---|---|---|---|
| **Agent A** | `feature/yahoo-api-client` | `src/api/yahoo_client.py`, Yahoo section of `src/db/loaders.py`, `tests/api/test_yahoo_client.py`, `tests/db/test_yahoo_loaders.py` | Authenticated Yahoo API client with normalized DataFrame outputs |
| **Agent B** | `feature/mlb-data-client` | `src/api/mlb_client.py`, MLB section of `src/db/loaders.py`, `tests/api/test_mlb_client.py`, `tests/db/test_mlb_loaders.py` | Call-up tracker, daily stats ingestion, projections loader |
| **Agent C** | `feature/analysis-modules` | `src/analysis/matchup_analyzer.py`, `src/analysis/waiver_ranker.py`, `src/analysis/lineup_optimizer.py`, all tests | Pure analysis functions with complete test coverage |
| **Agent D** | `feature/shiny-app` | `src/app/app.py`, `src/app/ui.py`, `src/app/server.py`, `src/app/stubs.py`, app tests | Full working UI with stub data, all four tabs functional |

### Why 4 (not more)

- Agent A and B are independent but both write to `src/db/loaders.py` — more agents
  would cause merge conflicts in that file
- Agent C has zero external dependencies (pure functions on DataFrames) — ideal for isolation
- Agent D can be built entirely on stub data, has no dependency on A, B, or C until integration
- A 5th agent for integration is tempting but integration requires human judgment on
  player ID mapping and OAuth token handling — keep that in main development

### Merge order for integration

```
main (Phase 1) → merge Agent C first (no conflicts, pure logic)
              → merge Agent D second (app + stubs, no conflicts)
              → merge Agent A third (Yahoo loaders)
              → merge Agent B fourth (MLB loaders, resolve any loader.py conflicts)
              → integration work begins
```

---

## Interface Contracts Summary

These are the DataFrame column guarantees that agents must honor.
All column names come from `src/db/schema.py` (single source of truth).

### fact_player_stats_daily (Agent B produces → Agent C consumes)
```
player_id, stat_date, ab, h, hr, sb, bb, errors, chances,
ip, w, k, walks_allowed, hits_allowed, sv, holds,
avg, ops, fpct, whip, k_bb, sv_h
```

### fact_projections (Agent B produces → Agent C consumes)
```
player_id, projection_date, target_week,
proj_h, proj_hr, proj_sb, proj_bb, proj_ip, proj_w, proj_k, proj_sv_h,
proj_avg, proj_ops, proj_whip, proj_k_bb, proj_fpct,
games_remaining, source
```

### matchup_analyzer output (Agent C produces → Agent D consumes)
```
category, my_value, opp_value, my_leads (bool),
margin_pct, win_prob, status
```

### waiver_ranker output (Agent C produces → Agent D consumes)
```
player_id, overall_score, category_scores (JSON),
is_callup (bool), days_since_callup,
recommended_drop_id, notes
```

### lineup_optimizer daily_report output (Agent C produces → Agent D consumes)
```python
{
  "lineup": {"C": "player_id", "1B": "player_id", ...},
  "adds": [{"add": "player_id", "drop": "player_id", "reason": str, "score": float}],
  "matchup_summary": pd.DataFrame,   # score_categories() output
  "ip_pace": {"current_ip": float, "projected_ip": float, "on_pace": bool},
  "callup_alerts": [{"player_id": str, "days_since_callup": int, "team": str}]
}
```

---

## Key Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Player ID mismatch (Yahoo ↔ MLB ↔ FanGraphs) | High | Use `pybaseball.playerid_lookup()` to build crosswalk on first load; log unmatched players for manual review |
| FanGraphs projection scraping blocked | High | Cache on first successful pull; tiered fallback to Statcast estimates then trailing-30-day stats |
| Yahoo access token expiry in GitHub Actions | Medium | Pipeline refreshes token on every run; write updated refresh token back to GitHub Secrets via API call |
| Rate stat projection math errors (AVG, WHIP) | Medium | Accumulate numerator/denominator separately (H+proj_H)/(AB+proj_AB) — never average rates directly |
| MLB Stats API (unofficial) endpoint changes | Medium | Pin endpoint URLs; monitor `python-mlb-statsapi` community library for breakage notices |
| Minor league stats gaps for newly called-up players | Medium | Fall back to MiLB.com stats or flag player as "limited data"; never block the pipeline on missing MiLB stats |
| shinyapps.io free tier exhaustion | Medium | Upgrade to Starter ($9/month) before season starts; app is read-only/fast so sessions are short |
| GitHub Actions cron drift (UTC vs MT) | Low | Set cron in UTC with comment noting MT equivalent; recheck at DST changeovers (Mar, Nov) |
| Yahoo API changes mid-season | Low | All Yahoo calls isolated in `yahoo_client.py`; one file to patch |
| Rotowire / third-party news sources | Resolved | Replaced with MLB Stats API for official IL/injury data — free, no ToS risk |
