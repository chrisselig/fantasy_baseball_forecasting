# Fantasy Baseball Intelligence App
## Project Description

### Overview
A Python-based fantasy baseball intelligence platform deployed on shinyapps.io that integrates
with Yahoo Fantasy Sports to deliver daily lineup recommendations, matchup projections, and
waiver wire intelligence — with a focus on real-time news and minor league call-up tracking.

---

### Core Modules

#### 1. Yahoo Fantasy API Integration Layer
The foundation of the entire application. Handles all authenticated communication with Yahoo's
Fantasy Sports API and normalizes data into a consistent internal format.

**Responsibilities:**
- OAuth 2.0 authentication and token refresh (Yahoo requires this)
- Fetch league settings, standings, and schedule
- Fetch current roster for user's team
- Fetch opponent rosters for current and upcoming matchups
- Fetch free agent / waiver wire pool with stats
- Fetch transaction history (adds, drops, trades)
- Cache responses to avoid rate limiting; refresh on a daily schedule

**Key data retrieved:**
- Player IDs, names, positions, eligibility
- Current week stats (accumulating totals for live matchup tracking)
- Season-to-date stats for all rostered and available players
- Injury status, news flags

---

#### 2. Matchup Analyzer
Projects the current week's head-to-head category outcomes and generates daily
actionable recommendations.

**Responsibilities:**
- Pull current week's accumulating stats for both teams
- Project end-of-week category totals based on remaining games and recent performance
- Score each of the 12 categories (win / loss / tie probability)
- Identify categories that are close (flippable) vs. locked (safe win or likely loss)
- Recommend daily lineup changes to target flippable categories
- Track IP accumulation vs. the 21 IP/week minimum and flag risk of penalty

**Category win logic:**
- Highest value wins: H, HR, SB, BB, FPCT, AVG, OPS, W, K, K/BB, SV+H
- Lowest value wins: WHIP

**Output:**
- Category-by-category scoreboard with projected final values
- Win probability per category
- "Focus categories" — where lineup decisions this week matter most
- Daily starting lineup recommendation

---

#### 3. Waiver Wire Intelligence Engine
Identifies the highest-value free agents available, with emphasis on recent call-ups
from the minor leagues and players with improving roles.

**Responsibilities:**
- Score all available free agents by projected category contribution
- Compare each player's value to the weakest contributor on the user's roster
  at the same position
- Rank add candidates by net category improvement
- Flag players with role changes, injuries to players ahead of them on the depth chart,
  or recent promotion from Triple-A / Double-A

**News & call-up data sources:**
- MLB.com transactions feed (official call-up / option / DFA records)
- Baseball Reference / FanGraphs recent stats and minor league splits
- Rotowire / MLB injury reports (scraped or via available APIs)
- Optional: Baseball Savant Statcast data for early breakout signals

**Call-up specific logic:**
- Cross-reference MLB transaction feed with Yahoo free agent pool
- Flag newly added players promoted within the last 7 days
- Pull minor league stats (AVG, HR, K%, BB%) as leading indicators
- Show MLB opportunity context (is the starter injured? platoon role? closer committee?)

**Output:**
- Ranked waiver wire targets with category impact scores
- "Call-up alert" section for recent promotions
- Side-by-side comparison of add candidate vs. player to drop

---

### Application Architecture

```
fantasy_baseball_forecasting/
├── config/
│   └── league_settings.yaml       # League configuration
├── src/
│   ├── api/
│   │   ├── yahoo_client.py        # Yahoo OAuth + API calls
│   │   └── mlb_client.py          # MLB transactions, Statcast, news feeds
│   ├── db/
│   │   ├── connection.py          # MotherDuck connection management
│   │   ├── schema.py              # Table creation and migrations
│   │   └── loaders.py             # ETL: API responses → MotherDuck tables
│   ├── analysis/
│   │   ├── matchup_analyzer.py    # Category projection and win probability
│   │   ├── waiver_ranker.py       # Free agent scoring and ranking
│   │   └── lineup_optimizer.py    # Daily lineup + add/drop recommendations
│   └── app/
│       ├── app.py                 # Shiny for Python app entry point
│       ├── ui.py                  # UI layout and components
│       └── server.py              # Reactive server logic
├── docs/
│   └── project_description.md
├── requirements.txt
└── README.md
```

---

### Data Storage: MotherDuck (Cloud DuckDB)

All ingested and processed data is persisted in **MotherDuck**, a cloud-hosted DuckDB
service. This gives the app fast analytical queries, a persistent store across sessions,
and a clean separation between data ingestion and the Shiny front-end.

**Connection:** `duckdb.connect("md:fantasy_baseball")` using a MotherDuck token stored
as an environment variable.

---

#### Schema Design

##### `dim_players`
Master player reference table. One row per player, updated daily.
```sql
CREATE TABLE dim_players (
    player_id       VARCHAR PRIMARY KEY,   -- Yahoo player key
    mlb_id          INTEGER,               -- MLB.com player ID for cross-referencing
    full_name       VARCHAR NOT NULL,
    team            VARCHAR,               -- MLB team abbreviation
    positions       VARCHAR[],             -- All eligible positions e.g. ['SS','2B']
    bats            CHAR(1),               -- L / R / S
    throws          CHAR(1),               -- L / R
    status          VARCHAR,               -- Active, IL-10, IL-60, Minors, NA
    updated_at      TIMESTAMP
);
```

##### `dim_dates`
Date spine used for joins and schedule alignment.
```sql
CREATE TABLE dim_dates (
    date            DATE PRIMARY KEY,
    season          INTEGER,
    week_number     INTEGER,              -- Fantasy week number
    is_playoff_week BOOLEAN,
    day_of_week     VARCHAR
);
```

##### `fact_player_stats_daily`
One row per player per date. Source of truth for all stat accumulation.
```sql
CREATE TABLE fact_player_stats_daily (
    player_id       VARCHAR,
    stat_date       DATE,
    -- Batter stats
    ab              INTEGER,
    h               INTEGER,
    hr              INTEGER,
    sb              INTEGER,
    bb              INTEGER,
    errors          INTEGER,
    chances         INTEGER,              -- For FPCT calculation
    -- Pitcher stats
    ip              DECIMAL(5,1),
    w               INTEGER,
    k               INTEGER,
    walks_allowed   INTEGER,
    hits_allowed    INTEGER,
    sv              INTEGER,
    holds           INTEGER,
    -- Computed on ingest
    avg             DECIMAL(5,3),
    ops             DECIMAL(5,3),
    fpct            DECIMAL(5,3),
    whip            DECIMAL(5,3),
    k_bb            DECIMAL(5,2),
    sv_h            INTEGER,
    PRIMARY KEY (player_id, stat_date),
    FOREIGN KEY (player_id) REFERENCES dim_players(player_id)
);
```

##### `fact_player_stats_weekly`
Aggregated weekly totals, rebuilt each Sunday. Used for matchup projections.
```sql
CREATE TABLE fact_player_stats_weekly (
    player_id       VARCHAR,
    week_number     INTEGER,
    season          INTEGER,
    -- Cumulative counting stats
    h               INTEGER,
    hr              INTEGER,
    sb              INTEGER,
    bb              INTEGER,
    ip              DECIMAL(6,1),
    w               INTEGER,
    k               INTEGER,
    sv_h            INTEGER,
    -- Rate stats (computed from aggregated components)
    avg             DECIMAL(5,3),
    ops             DECIMAL(5,3),
    fpct            DECIMAL(5,3),
    whip            DECIMAL(5,3),
    k_bb            DECIMAL(5,2),
    PRIMARY KEY (player_id, week_number, season)
);
```

##### `fact_matchups`
One row per fantasy matchup (team vs. team, per week).
```sql
CREATE TABLE fact_matchups (
    matchup_id      VARCHAR PRIMARY KEY,  -- e.g. '87941_2026_W01_T3vsT7'
    league_id       INTEGER,
    week_number     INTEGER,
    season          INTEGER,
    team_id_home    VARCHAR,
    team_id_away    VARCHAR,
    -- Category results (populated at end of week)
    h_home          INTEGER,   h_away    INTEGER,
    hr_home         INTEGER,   hr_away   INTEGER,
    sb_home         INTEGER,   sb_away   INTEGER,
    bb_home         INTEGER,   bb_away   INTEGER,
    avg_home        DECIMAL(5,3), avg_away DECIMAL(5,3),
    ops_home        DECIMAL(5,3), ops_away DECIMAL(5,3),
    fpct_home       DECIMAL(5,3), fpct_away DECIMAL(5,3),
    w_home          INTEGER,   w_away    INTEGER,
    k_home          INTEGER,   k_away    INTEGER,
    whip_home       DECIMAL(5,3), whip_away DECIMAL(5,3),
    k_bb_home       DECIMAL(5,2), k_bb_away DECIMAL(5,2),
    sv_h_home       INTEGER,   sv_h_away INTEGER,
    -- Outcome
    categories_won_home  INTEGER,
    categories_won_away  INTEGER,
    result          VARCHAR    -- 'home_win', 'away_win', 'tie'
);
```

##### `fact_rosters`
Snapshot of each team's roster by date. Tracks adds/drops/trades over time.
```sql
CREATE TABLE fact_rosters (
    team_id         VARCHAR,
    player_id       VARCHAR,
    snapshot_date   DATE,
    roster_slot     VARCHAR,              -- C, 1B, SP, BN, IL, NA, etc.
    acquisition_type VARCHAR,             -- 'draft', 'waiver', 'trade', 'fa'
    PRIMARY KEY (team_id, player_id, snapshot_date)
);
```

##### `fact_transactions`
Every add, drop, and trade in the league.
```sql
CREATE TABLE fact_transactions (
    transaction_id  VARCHAR PRIMARY KEY,
    league_id       INTEGER,
    transaction_date TIMESTAMP,
    type            VARCHAR,              -- 'add', 'drop', 'trade'
    team_id         VARCHAR,
    player_id       VARCHAR,
    from_team_id    VARCHAR,             -- NULL for add from FA
    notes           VARCHAR
);
```

##### `fact_waiver_scores`
Daily scoring of free agents by the waiver intelligence engine.
```sql
CREATE TABLE fact_waiver_scores (
    player_id           VARCHAR,
    score_date          DATE,
    overall_score       DECIMAL(8,4),
    category_scores     JSON,            -- per-category contribution scores
    is_callup           BOOLEAN,
    days_since_callup   INTEGER,
    recommended_drop_id VARCHAR,         -- player_id of suggested drop
    notes               VARCHAR,
    PRIMARY KEY (player_id, score_date)
);
```

##### `fact_projections`
Forward-looking projections by player by week, refreshed daily.
```sql
CREATE TABLE fact_projections (
    player_id       VARCHAR,
    projection_date DATE,
    target_week     INTEGER,
    -- Projected counting stats for remaining days in week
    proj_h          DECIMAL(6,2),
    proj_hr         DECIMAL(6,2),
    proj_sb         DECIMAL(6,2),
    proj_bb         DECIMAL(6,2),
    proj_ip         DECIMAL(6,2),
    proj_w          DECIMAL(6,2),
    proj_k          DECIMAL(6,2),
    proj_sv_h       DECIMAL(6,2),
    proj_avg        DECIMAL(5,3),
    proj_ops        DECIMAL(5,3),
    proj_whip       DECIMAL(5,3),
    proj_k_bb       DECIMAL(5,2),
    proj_fpct       DECIMAL(5,3),
    games_remaining INTEGER,
    source          VARCHAR,             -- 'steamer', 'zips', 'internal'
    PRIMARY KEY (player_id, projection_date, target_week)
);
```

---

#### Data Flow

```
Yahoo API  ──►  yahoo_client.py  ──►  MotherDuck (fact_rosters, fact_transactions)
MLB API    ──►  mlb_client.py    ──►  MotherDuck (dim_players, fact_player_stats_daily)
FanGraphs  ──►  projections.py   ──►  MotherDuck (fact_projections)
                                            │
                                            ▼
                              analysis/ modules query MotherDuck
                                            │
                                            ▼
                                    Shiny app renders results
```

---

### Deployment: Shiny for Python on shinyapps.io

The app will be built using **Shiny for Python** (`shiny` package) and deployed to
shinyapps.io via `rsconnect-python`.

**Daily refresh strategy:**
- App triggers a data refresh on first load each day (or on a schedule via a background task)
- Yahoo API token stored as an environment variable / secret on shinyapps.io
- Cached data stored between sessions to minimize API calls

**Key UI views:**
1. **Dashboard** — Daily command center:
   - Recommended starting lineup for today (who to start/sit at each position)
   - Projected adds for today and the rest of the week, ranked by category impact
   - Projected drops — weakest contributors given the week's remaining schedule
   - Live matchup category scoreboard with current standings and win/loss probability
   - IP tracker vs. 21-inning minimum with pace indicator
   - Call-up alerts — newly promoted players available on waivers
2. **Matchup** — Full category breakdown with win probabilities and projections
3. **Waiver Wire** — Ranked add targets, call-up alerts, drop candidates
4. **Roster** — Current team stats by category, strength/weakness summary

---

### External Dependencies

| Purpose | Library / Source |
|---|---|
| Yahoo API auth | `yahoo_oauth`, `requests` |
| Data manipulation | `pandas`, `numpy` |
| MLB transactions / news | `requests`, `beautifulsoup4` |
| Statcast data | `pybaseball` |
| Shiny app framework | `shiny` (Shiny for Python) |
| Deployment | `rsconnect-python` |
| Scheduling / caching | `apscheduler`, `diskcache` |
| Data warehouse | `duckdb` + MotherDuck cloud |

---

### Success Criteria
- Daily lineup card generated before first game each day
- Matchup win probability updated with live accumulating stats
- Waiver wire call-up alerts surfaced within 24 hours of promotion
- App loads and refreshes within shinyapps.io free tier constraints
