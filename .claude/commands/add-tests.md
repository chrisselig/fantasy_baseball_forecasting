# Add Tests Command

Write comprehensive tests for: $ARGUMENTS

## Step 1 — Analyze the target

Read $ARGUMENTS and identify:
- All public functions and classes
- External dependencies (Yahoo API, MLB API, MotherDuck, `pybaseball`) — these must be mocked
- Side effects (DB writes, API calls, file I/O)
- Any existing tests in `tests/` for this module

Determine which layer the code belongs to:
- **`src/api/`** — API client code (mock HTTP responses)
- **`src/db/`** — Database layer (use in-memory DuckDB, not MotherDuck)
- **`src/analysis/`** — Analysis/logic layer (mock DB queries, use fixture DataFrames)
- **`src/app/`** — Shiny app layer (test server logic in isolation from UI)

---

## Step 2 — Create the test file

Place tests at `tests/<matching_subpath>/test_<filename>.py` mirroring the `src/` structure.

Use `pytest`. Structure each test file as:
```python
# tests/<subpath>/test_<module>.py

import pytest
import pandas as pd
# ... imports

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_<entity>() -> pd.DataFrame:
    """Minimal valid DataFrame for <entity> tests."""
    ...

# ── Happy path ───────────────────────────────────────────────────────────────

def test_<function>_returns_expected_output(...):
    ...

# ── Edge cases ───────────────────────────────────────────────────────────────

def test_<function>_handles_empty_input(...):
    ...

# ── Failure cases ────────────────────────────────────────────────────────────

def test_<function>_raises_on_invalid_input(...):
    ...
```

---

## Step 3 — Coverage requirements

For every public function, write tests that cover:

| Case | Required |
|---|---|
| Happy path with realistic data | Yes |
| Empty DataFrame / no results | Yes |
| Missing or null required columns | Yes |
| Boundary values (0 IP, .000 AVG, etc.) | Where applicable |
| API/DB error paths (mocked exceptions) | For api/ and db/ modules |
| WHIP lower-is-better logic | For any matchup/scoring functions |
| IP minimum enforcement (21 IP/week) | For lineup/scheduling functions |

---

## Step 4 — Mocking conventions

- Use `pytest-mock` (`mocker` fixture) or `unittest.mock.patch`
- Mock Yahoo API calls at the `requests.get` / `requests.post` level
- Mock MotherDuck with an **in-memory DuckDB** connection: `duckdb.connect(":memory:")`
- Never make real network calls in tests
- Store reusable mock responses as JSON fixtures in `tests/fixtures/`

---

## Step 5 — Run and verify

```bash
pytest tests/<path>/test_<module>.py -v
pytest --cov=src/<module_path> --cov-report=term-missing
```

All tests must pass. Coverage for the target module should be ≥ 80%.

---

## Output

After writing tests, summarize:
```
Tests written for: $ARGUMENTS
Test file: tests/<path>/test_<module>.py
Functions covered: [list]
Cases tested: [N] happy path, [N] edge cases, [N] failure cases
Mocks used: [list external dependencies mocked]
Coverage estimate: ~[N]%
Run with: pytest tests/<path>/test_<module>.py -v
```
