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
│       └── daily_pipeline.yml     # Cron pipeline: runs daily at 9am MT
├── config/
│   └── league_settings.yaml       # League configuration (scoring, roster, schedule)
├── scripts/
│   └── yahoo_auth.py              # One-time local OAuth token generator
├── src/
│   ├── api/
│   │   ├── yahoo_client.py        # Yahoo OAuth + API calls
│   │   └── mlb_client.py          # MLB Stats API, Statcast, minor league data
│   ├── db/
│   │   ├── connection.py          # MotherDuck connection management
│   │   ├── schema.py              # Table creation and migrations
│   │   └── loaders.py             # ETL: API responses → MotherDuck tables
│   ├── analysis/
│   │   ├── matchup_analyzer.py    # Category projection and win probability
│   │   ├── waiver_ranker.py       # Free agent scoring and ranking
│   │   └── lineup_optimizer.py    # Daily lineup + add/drop recommendations
│   ├── pipeline/
│   │   └── daily_run.py           # GitHub Actions entry point
│   └── app/
│       ├── app.py                 # Shiny for Python app entry point
│       ├── ui.py                  # UI layout and components
│       ├── server.py              # Reactive server logic (read-only from MotherDuck)
│       └── stubs.py               # Mock data for dev/offline fallback
├── tests/                         # Unit and integration tests
├── docs/
│   └── project_description.md    # Full project description and schema design
├── .venv/                         # Virtual environment (untracked)
├── pyproject.toml                 # Project metadata and dependencies
├── requirements.txt               # Deployment dependencies for shinyapps.io
├── README.md
└── .gitignore
```

## Important
1. Before making any change, create and checkout a feature branch: `feature/short-description`
2. Write automated tests for all code.
3. All tests must pass before committing.
4. Submit a pull request for code review before merging into main.
5. Install all Python libraries to the virtual environment.

## Development Standards
- **Language:** Python 3.12
- **Code style:** Follow PEP 8 strictly, use Black for formatting
- **Type hints:** Required for all function signatures and class definitions
- **Docstrings:** Required for all public functions and classes

## Workflow Requirements
1. Create feature branch: `feature/[short-description]`
2. Write unit tests for all data processing and analysis functions
3. Run `pytest` and ensure all tests pass
4. Run `black .` and `flake8` before committing
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
