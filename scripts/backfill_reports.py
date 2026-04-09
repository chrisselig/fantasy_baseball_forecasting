"""
backfill_reports.py

Backfill daily reports for a range of dates.

Re-runs the analysis step (matchup projection + category scoring) for
each date and writes the result to fact_daily_reports.  This lets the
app's week selector show historical weeks.

Prerequisites:
  - fact_player_stats_daily must already be populated for the date range
    (run ``backfill_stats.py`` first).
  - fact_rosters and fact_matchups should contain data for the relevant
    weeks (populated by the daily pipeline's Yahoo step).

Usage::

    python -m scripts.backfill_reports 2026-03-25 2026-04-07

Or from GitHub Actions via the backfill_reports workflow.
"""

from __future__ import annotations

import datetime
import logging
import sys
import uuid

import pandas as pd

from src.config import load_league_settings
from src.db.connection import managed_connection
from src.db.loaders_mlb import get_fantasy_week
from src.db.schema import create_all_tables
from src.pipeline.daily_run import (
    _get_season_start,
    _step_refresh_projections,
    _step_run_analysis,
    _step_write_daily_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def backfill_reports(start_date: datetime.date, end_date: datetime.date) -> None:
    """Generate daily reports for each date in [start_date, end_date].

    For each date the script:
      1. Determines the fantasy week.
      2. Refreshes pace-based projections for that date.
      3. Runs the analysis step (matchup scoring, IP pace).
      4. Writes the report to fact_daily_reports.

    Free-agent ranking and lineup optimization are skipped for historical
    dates since the data is no longer actionable.

    Args:
        start_date: First date to generate a report for (inclusive).
        end_date: Last date to generate a report for (inclusive).
    """
    settings = load_league_settings()
    reports_written = 0
    days = 0

    with managed_connection() as conn:
        create_all_tables(conn)

        current = start_date
        while current <= end_date:
            season = current.year
            week = get_fantasy_week(current, _get_season_start(season))
            run_id = str(uuid.uuid4())

            logger.info("Generating report for %s (week %d) ...", current, week)

            try:
                # Refresh projections for this date so analysis has data
                _step_refresh_projections(conn, current, week, season)

                # Run analysis with empty free-agent and schedule DataFrames
                # (lineup/waiver steps will produce empty results, which is fine)
                empty_fa = pd.DataFrame(columns=["player_id"])
                empty_schedule = pd.DataFrame(columns=["player_id"])

                report = _step_run_analysis(
                    conn,
                    today=current,
                    week=week,
                    season=season,
                    settings=settings,
                    fa_df=empty_fa,
                    schedule_df=empty_schedule,
                )

                if report:
                    _step_write_daily_report(
                        conn, report, current, week, season, run_id
                    )
                    reports_written += 1
                    logger.info("  %s: report written (week %d).", current, week)
                else:
                    logger.warning("  %s: analysis returned empty report.", current)

            except Exception as exc:
                logger.error("  %s: failed — %s", current, exc)

            days += 1
            current += datetime.timedelta(days=1)

    logger.info(
        "Report backfill complete: %d days processed, %d reports written.",
        days,
        reports_written,
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m scripts.backfill_reports START_DATE END_DATE")
        print("  Dates in YYYY-MM-DD format.")
        print("  Example: python -m scripts.backfill_reports 2026-03-25 2026-04-07")
        sys.exit(1)

    start = datetime.date.fromisoformat(sys.argv[1])
    end = datetime.date.fromisoformat(sys.argv[2])

    if start > end:
        print(f"Error: start_date ({start}) is after end_date ({end})")
        sys.exit(1)

    backfill_reports(start, end)
