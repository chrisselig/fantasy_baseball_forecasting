"""
backfill_stats.py

Backfill daily MLB stats for a range of dates.

Runs the MLB stats loading step of the daily pipeline for each date
in the given range. Does NOT re-run Yahoo or analysis steps — just
loads batter and pitcher stats from the MLB Stats API into
fact_player_stats_daily.

Usage::

    python -m scripts.backfill_stats 2026-03-25 2026-04-07

Or from GitHub Actions via the backfill_stats workflow.
"""

from __future__ import annotations

import datetime
import logging
import sys

from src.db.connection import managed_connection
from src.db.schema import create_all_tables
from src.pipeline.daily_run import _step_load_mlb_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def backfill(start_date: datetime.date, end_date: datetime.date) -> None:
    """Load MLB stats for each date in [start_date, end_date].

    Args:
        start_date: First date to backfill (inclusive).
        end_date: Last date to backfill (inclusive).
    """
    total_rows = 0
    days = 0

    with managed_connection() as conn:
        create_all_tables(conn)

        current = start_date
        while current <= end_date:
            logger.info("Backfilling stats for %s ...", current)
            try:
                # _step_load_mlb_stats fetches stats for current-1 (yesterday)
                # so we pass current+1 to get stats for current
                fetch_date = current + datetime.timedelta(days=1)
                row_counts, _ = _step_load_mlb_stats(
                    conn, fetch_date, week=0, season=current.year
                )
                day_total = sum(row_counts.values())
                total_rows += day_total
                logger.info(
                    "  %s: %d rows written (%s)",
                    current,
                    day_total,
                    row_counts,
                )
            except Exception as exc:
                logger.error("  %s: failed — %s", current, exc)

            days += 1
            current += datetime.timedelta(days=1)

    logger.info(
        "Backfill complete: %d days processed, %d total rows written.",
        days,
        total_rows,
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m scripts.backfill_stats START_DATE END_DATE")
        print("  Dates in YYYY-MM-DD format.")
        print("  Example: python -m scripts.backfill_stats 2026-03-25 2026-04-07")
        sys.exit(1)

    start = datetime.date.fromisoformat(sys.argv[1])
    end = datetime.date.fromisoformat(sys.argv[2])

    if start > end:
        print(f"Error: start_date ({start}) is after end_date ({end})")
        sys.exit(1)

    backfill(start, end)
