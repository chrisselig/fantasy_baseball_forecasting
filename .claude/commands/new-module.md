# New Module Command

Scaffold a new Python module for: $ARGUMENTS

## Step 1 — Classify the module

Determine which layer $ARGUMENTS belongs to and where it should live:

| Layer | Path | Purpose |
|---|---|---|
| API client | `src/api/` | Fetching data from Yahoo, MLB, or external sources |
| Database | `src/db/` | MotherDuck schema, loaders, or query helpers |
| Analysis | `src/analysis/` | Scoring, projection, ranking, optimization logic |
| App | `src/app/` | Shiny UI, server, or reactive components |

---

## Step 2 — Create the module file

Create `src/<layer>/<module_name>.py` with this structure:

```python
"""
<module_name>.py

<One paragraph describing what this module does, its inputs, and its outputs.>
"""

from __future__ import annotations

import logging
from typing import ...

import pandas as pd
# ... other imports

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

# Any module-level constants go here


# ── Public API ───────────────────────────────────────────────────────────────

def main_function(param: Type) -> ReturnType:
    """
    Short description of what this function does.

    Args:
        param: Description of the parameter.

    Returns:
        Description of the return value.

    Raises:
        ValueError: When input is invalid.
    """
    ...


# ── Private helpers ──────────────────────────────────────────────────────────

def _helper(param: Type) -> ReturnType:
    ...
```

---

## Step 3 — Layer-specific requirements

### If `src/api/`:
- Accept a `requests.Session` or connection object as a parameter (don't create it internally)
- Return normalized `pd.DataFrame` or typed dataclass — never raw JSON
- Raise a descriptive exception on HTTP errors (not silent failure)
- Log request URLs and response codes at DEBUG level

### If `src/db/`:
- Accept a `duckdb.DuckDBPyConnection` as a parameter
- Return `pd.DataFrame` for reads, `int` (row count) for writes
- Use parameterized queries — no f-string SQL
- Follow the existing `dim_` / `fact_` naming conventions

### If `src/analysis/`:
- Accept DataFrames as inputs; return DataFrames or typed results
- Keep business logic pure — no API calls or DB connections inside analysis functions
- Remember WHIP is **lowest wins**; all other categories are highest wins
- Account for the 21 IP/week minimum in any pitching-related logic

### If `src/app/`:
- Keep reactive expressions, outputs, and data fetching clearly separated
- Cache expensive computations with `@reactive.calc`
- Do not call analysis functions directly from UI callbacks — go through the server layer

---

## Step 4 — Create a test file

Create `tests/<layer>/test_<module_name>.py` with at least:
- One happy-path test per public function
- One edge-case test (empty input, zero values, etc.)
- Mocks for any external dependencies (API, DB)

---

## Step 5 — Register the module

- If `src/api/`: import and call from `src/db/loaders.py` where appropriate
- If `src/db/`: add any new tables to `src/db/schema.py`
- If `src/analysis/`: import into `src/app/server.py` where the output will be used
- Update `docs/project_description.md` with a brief description of the new module

---

## Output

```
Module created: src/<layer>/<module_name>.py
Test file created: tests/<layer>/test_<module_name>.py
Layer: [API / DB / Analysis / App]
Public functions: [list]
Registered in: [file where it was imported/wired up]
Docs updated: [yes / no]
```
