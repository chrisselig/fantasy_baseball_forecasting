# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python-based fantasy baseball intelligence platform for the Yahoo Fantasy Baseball league
"The Vlad Guerrero Invitational" (League ID: 87941). The app integrates with the Yahoo Fantasy
Sports API, ingests MLB stats and news, stores everything in MotherDuck (cloud DuckDB), and
surfaces daily lineup recommendations, matchup projections, and waiver wire intelligence via
a Shiny for Python app deployed on shinyapps.io.

Full project description: `docs/project_description.md`
League settings: `config/league_settings.yaml`

## Expected Stack

- **Language:** Python 3.12
- **IDE:** VS Code
- **Data warehouse:** MotherDuck (cloud DuckDB) — connect via `duckdb.connect("md:fantasy_baseball")`
- **App framework:** Shiny for Python (`shiny`)
- **Deployment:** shinyapps.io via `rsconnect-python`
- **Core packages:** `yahoo_oauth`, `requests`, `pandas`, `numpy`, `duckdb`, `pybaseball`, `shiny`
- **Pipeline scheduling:** GitHub Actions (cron) — not APScheduler; the app is read-only on shinyapps.io

## Project Structure

```
fantasy_baseball_forecasting/
├── .github/
│   └── workflows/
│       ├── ci.yml                 # CI (ruff + mypy + pytest) on PRs; deploy to shinyapps.io on push to main
│       ├── daily_pipeline.yml     # Cron pipeline: runs daily at 9am MT
│       ├── backfill_stats.yml     # Manual: backfill daily MLB stats for a date range
│       ├── seed_data.yml          # Manual: seed MotherDuck with a 2025 simulation dataset
│       └── debug_projections.yml  # Manual: dump projection debug output for troubleshooting
├── .claude/
│   └── commands/                  # Project slash commands (see "Project Commands" below)
├── config/
│   └── league_settings.yaml       # League configuration (scoring, roster, schedule)
├── scripts/
│   ├── yahoo_auth.py              # One-time local OAuth token generator
│   ├── backfill_reports.py        # Regenerate fact_daily_reports for a date range
│   ├── backfill_stats.py          # Load fact_player_stats_daily for a date range
│   ├── cleanup_bad_weeks.py       # Remove malformed/partial week rows from the warehouse
│   ├── debug_projections.py       # Print projection internals for a given date/player
│   ├── seed_2025_test.py          # Seed a small 2025 fixture dataset for local/dev testing
│   └── seed_motherduck_2025.py    # Seed a full 2025 simulation dataset into MotherDuck
├── src/
│   ├── config.py                 # Load league_settings.yaml + env config helpers
│   ├── api/
│   │   ├── yahoo_client.py        # Yahoo OAuth + API calls
│   │   └── mlb_client.py          # MLB Stats API, Statcast, minor league data
│   ├── db/
│   │   ├── connection.py          # MotherDuck connection management
│   │   ├── schema.py              # Table creation and migrations
│   │   ├── loaders_yahoo.py       # ETL: Yahoo API responses → MotherDuck tables
│   │   ├── loaders_mlb.py         # ETL: MLB stats/schedule → MotherDuck tables
│   │   ├── loaders_news.py        # ETL: player news → fact_player_news
│   │   └── loaders_advanced.py    # ETL: Statcast/advanced metrics → fact_player_advanced_stats
│   ├── analysis/
│   │   ├── matchup_analyzer.py    # Category projection and win probability
│   │   ├── waiver_ranker.py       # Free agent scoring and ranking
│   │   ├── lineup_optimizer.py    # Daily lineup + add/drop recommendations
│   │   ├── hot_cold.py            # Rolling hot/cold streak detection
│   │   ├── news.py                # News sentiment/impact tagging
│   │   └── shrinkage.py           # Regression-to-mean shrinkage for small-sample stats
│   ├── pipeline/
│   │   ├── daily_run.py           # GitHub Actions entry point
│   │   └── token_refresh.py       # Rotate + write back Yahoo OAuth tokens
│   └── app/
│       ├── app.py                 # Shiny for Python app entry point
│       ├── ui.py                  # UI layout and components
│       ├── server.py              # Reactive server logic (read-only from MotherDuck)
│       └── stubs.py               # Mock data for dev/offline fallback
├── tests/                         # Unit and integration tests
├── docs/
│   └── project_description.md    # Full project description and schema design (canonical doc)
├── app.py                         # shinyapps.io entry shim: loads MOTHERDUCK_TOKEN (from
│                                  #   _deploy_config.py in CI, else .env) then re-exports src/app/app.py
├── .venv/                         # Virtual environment (untracked)
├── pyproject.toml                 # Project metadata and dependencies
├── requirements.txt               # Deployment dependencies for shinyapps.io
├── README.md                      # Project overview and quick-start
└── .gitignore
```

## Project Commands

Project-specific slash commands live in `.claude/commands/`:

- `add-tests` — scaffold tests for a module
- `backfill` — guided historical stats/reports backfill for a date range
- `code-conventions` — project code style and conventions reference
- `code-review` — structured code review checklist
- `debug-api` — troubleshoot Yahoo/MLB API calls
- `deploy` — pre-flight checklist and shinyapps.io deploy
- `document-feature` — document a feature
- `new-db-table` — scaffold a new MotherDuck table (schema + loader)
- `new-module` — scaffold a new source module with tests
- `pipeline-health` — verify the daily pipeline landed and data is fresh

## Important
1. Before making any change, create and checkout a feature branch: `feature/short-description`
2. Write automated tests for all code.
3. All tests must pass before committing.
4. Submit a pull request for code review before merging into main.
5. Install all Python libraries to the virtual environment.

## Development Standards
- **Language:** Python 3.12
- **Code style:** PEP 8, enforced by `ruff format` and `ruff check`
- **Type hints:** Required for all function signatures and class definitions
- **Docstrings:** Required for all public functions and classes

## Workflow Requirements
1. Create feature branch: `feature/[short-description]`
2. Write unit tests for all data processing and analysis functions
3. Run `pytest` and ensure all tests pass
4. Run `ruff format .` and `ruff check .` and `mypy .` before committing
   (`mypy .` type-checks the whole repo — `src/`, `tests/`, and `scripts/` — all must be clean)
5. Update `docs/project_description.md` if adding or changing a module

## Data Handling Standards
- All API keys and tokens (Yahoo OAuth, MotherDuck token) must be stored as environment variables — never hardcoded
- Never commit credentials, `.env` files, or token files to version control
- Use `duckdb` for all data persistence — no local CSV or pickle files as a data store
- Prefer vectorized Pandas operations over row-by-row iteration
- Rate limit Yahoo API calls; use local cache for within-session repeated reads

## Dependencies
- **API / data:** `yahoo_oauth`, `requests`, `pybaseball`
- **Data processing:** `pandas`, `numpy`
- **Database:** `duckdb`
- **App / deployment:** `shiny`, `rsconnect-python`
- **Pipeline scheduling:** GitHub Actions (no APScheduler — app is read-only)
- **Testing:** `pytest`, `pytest-cov`
- **Formatting / linting:** `ruff`, `mypy`
