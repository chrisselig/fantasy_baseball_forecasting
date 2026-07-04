# Pipeline Health Command

Check whether the daily pipeline actually landed and the MotherDuck data is
fresh. Optional argument `$ARGUMENTS` is a date (YYYY-MM-DD) to check against;
if omitted, use today (in MT).

The daily pipeline runs at 9am MT via `.github/workflows/daily_pipeline.yml`
and records every run in `fact_pipeline_runs`. A healthy run has
`status = 'success'`, non-empty `rows_written`, and updates the key fact
tables for the target date.

---

## Step 1 — Connect to MotherDuck

`MOTHERDUCK_TOKEN` must be set in the environment. Connect read-only:

```python
import duckdb

conn = duckdb.connect("md:fantasy_baseball")
```

If the connection fails (missing token, network), skip to Step 5 and use the
GitHub Actions fallback.

---

## Step 2 — Inspect the latest pipeline run

Query the most recent row in `fact_pipeline_runs`:

```sql
SELECT run_id, run_at, status, rows_written, errors, duration_seconds
FROM fact_pipeline_runs
ORDER BY run_at DESC
LIMIT 5;
```

Report the newest run in a table:

| Field | Value |
|---|---|
| `run_at` | timestamp of the run |
| `status` | success / partial / failed |
| `duration_seconds` | how long it took |
| `rows_written` | per-table row counts (JSON) |
| `errors` | error text if `status != 'success'` |

Flags to raise:
- **Stale**: `run_at` is not from today (or the date in `$ARGUMENTS`) — the
  schedule may not have fired, or `PIPELINE_ENABLED` is not `'true'`.
- **Failed / partial**: surface the `errors` field verbatim.
- **Empty write**: `rows_written` is null or all zeros.

---

## Step 3 — Check freshness of the key fact tables

Confirm the tables the app reads were actually updated for the target date:

```sql
-- Daily player stats
SELECT MAX(stat_date) AS latest_stat_date, COUNT(*) AS rows_latest
FROM fact_player_stats_daily
WHERE stat_date = (SELECT MAX(stat_date) FROM fact_player_stats_daily);

-- Daily reports (drives the app's week selector)
SELECT MAX(report_date) AS latest_report_date, season, week_number
FROM fact_daily_reports
GROUP BY season, week_number
ORDER BY latest_report_date DESC
LIMIT 1;

-- Matchups (current week)
SELECT season, MAX(week_number) AS latest_week, COUNT(*) AS matchup_rows
FROM fact_matchups
GROUP BY season
ORDER BY season DESC
LIMIT 1;
```

Present the results as a freshness table:

| Table | Latest date / week | Rows | Fresh? |
|---|---|---|---|
| `fact_player_stats_daily` | `latest_stat_date` | count | ✓ if == target date |
| `fact_daily_reports` | `latest_report_date` | — | ✓ if == target date |
| `fact_matchups` | `season` / `latest_week` | count | ✓ if current week present |

A table whose latest date lags the target date by more than a day is stale —
call it out.

---

## Step 4 — Diagnose

- **Run succeeded but tables stale** → a specific step failed silently; read
  `errors` and cross-check `rows_written` for which tables got zero rows.
- **No recent run row at all** → the workflow did not fire. Check that
  `PIPELINE_ENABLED == 'true'` and the cron schedule, then Step 5.
- **`status = 'partial'`** → Yahoo or MLB fetch degraded; the app still works
  on last-good data but flag which categories are missing.

---

## Step 5 — GitHub Actions fallback

If MotherDuck is unreachable, or to corroborate the DB state, check the
workflow run history directly:

```bash
gh run list --workflow=daily_pipeline.yml --limit 5
gh run view <run-id> --log-failed   # inspect a failed run
```

Cross-reference the latest Actions run's conclusion (success/failure) and
timestamp against what `fact_pipeline_runs` reports.

---

## Output

```
Pipeline health for <target-date>:

Latest run:   <run_at>  —  status=<status>  (<duration>s)
Rows written: <summary of rows_written>
Errors:       <none | error text>

Freshness:
  fact_player_stats_daily  →  <latest_stat_date>   <✓ fresh | ✗ stale>
  fact_daily_reports       →  <latest_report_date> <✓ fresh | ✗ stale>
  fact_matchups            →  <season> W<week>      <✓ fresh | ✗ stale>

Verdict: HEALTHY / DEGRADED / FAILED / DID-NOT-RUN
Action:  [what to do next, if anything]
```
