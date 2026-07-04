# Backfill Command

Guided historical backfill for a date range. Argument `$ARGUMENTS` is the range
to backfill, e.g. `2026-03-25 2026-04-07` (start and end dates, inclusive).

There are two backfill scripts and they must run **in order**:

1. `scripts/backfill_stats.py` — loads daily MLB batter/pitcher stats into
   `fact_player_stats_daily`.
2. `scripts/backfill_reports.py` — regenerates `fact_daily_reports` (matchup
   projection + category scoring) so the app's week selector shows historical
   weeks. Depends on the stats being loaded first.

---

## Step 1 — Parse and validate the range

From `$ARGUMENTS`, extract `START_DATE` and `END_DATE` in `YYYY-MM-DD` form.
Both are inclusive. Sanity-check before running anything:

- `START_DATE <= END_DATE` (the scripts exit 1 otherwise).
- The range is within a real MLB season for the year(s) involved.
- A very wide range (many weeks) will make a lot of MLB Stats API calls — warn
  the user and confirm before proceeding.

`MOTHERDUCK_TOKEN` must be set in the environment (both scripts use
`managed_connection()` → `md:fantasy_baseball`).

---

## Step 2 — Backfill daily stats

Run from the repo root (module form, so `src` imports resolve):

```bash
python -m scripts.backfill_stats START_DATE END_DATE
```

What it does per day in the range:
- Calls `_step_load_mlb_stats` for that date (fetching the day's box-score
  stats) and upserts into `fact_player_stats_daily`.
- Logs rows written per day; a failed day is logged and skipped (the loop
  continues) rather than aborting the whole range.

Confirm from the final log line: `N days processed, M total rows written`.
If a day wrote 0 rows, check whether games were actually played that date.

---

## Step 3 — Backfill daily reports

Only after Step 2 succeeds for the range:

```bash
python -m scripts.backfill_reports START_DATE END_DATE
```

What it does per day:
1. Determines the fantasy week (`get_fantasy_week`).
2. Refreshes pace-based projections for the date.
3. Runs the analysis step (matchup scoring, IP pace).
4. Writes the report to `fact_daily_reports`.

Free-agent ranking and lineup optimization are intentionally **skipped** for
historical dates (the data is no longer actionable) — empty FA/schedule frames
are passed in, so empty lineup/waiver output is expected and fine.

Prerequisite reminder: `fact_rosters` and `fact_matchups` should already hold
data for the weeks in range (populated by the daily pipeline's Yahoo step). If
they don't, reports will be thin.

---

## Step 4 — Verify

Confirm the backfilled range landed:

```sql
-- Stats coverage
SELECT stat_date, COUNT(*) AS rows
FROM fact_player_stats_daily
WHERE stat_date BETWEEN 'START_DATE' AND 'END_DATE'
GROUP BY stat_date
ORDER BY stat_date;

-- Report coverage
SELECT report_date, season, week_number
FROM fact_daily_reports
WHERE report_date BETWEEN 'START_DATE' AND 'END_DATE'
ORDER BY report_date;
```

Every date with MLB games should appear in both. Gaps mean a day failed —
re-run just that narrower range.

---

## Safety notes

- **Idempotent.** Both scripts upsert (`INSERT OR REPLACE`) keyed by
  `(player_id, stat_date)` / `report_date`, so re-running the same range is
  safe and simply overwrites with fresh values. Prefer re-running over any
  manual deletion.
- **Never delete unbounded.** Do not `DELETE` from these tables without a tight
  `WHERE` bound on `season`/`week_number` (or `stat_date`/`report_date`). An
  unbounded delete can wipe live current-season data.
- **Order matters.** Always run `backfill_stats` before `backfill_reports` —
  reports read the stats table.
- **Watch the range size.** Backfilling months at once hits the MLB Stats API
  hard; chunk into week-sized ranges if you hit rate limits, and re-run failed
  chunks (idempotency makes this safe).

---

## Output

```
Backfill range: START_DATE → END_DATE

Stats:   <N days, M rows>   (gaps: <dates or none>)
Reports: <N reports written> (gaps: <dates or none>)

Verdict: COMPLETE / PARTIAL (re-run: <ranges>)
```
