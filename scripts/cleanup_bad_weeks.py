"""
cleanup_bad_weeks.py

Remove invalid reports from fact_daily_reports.
Deletes rows outside the valid range (1-26) AND rows from old seasons
(e.g. 2025 test data showing as Week 12).

Usage::

    python -m scripts.cleanup_bad_weeks
"""

from __future__ import annotations

import datetime
import logging

from src.db.connection import managed_connection
from src.db.schema import FACT_DAILY_REPORTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# Current season year
_CURRENT_SEASON = datetime.date.today().year


def cleanup_bad_weeks() -> None:
    """Delete rows from fact_daily_reports with invalid data."""
    with managed_connection() as conn:
        # Find bad rows: wrong week range OR wrong season
        bad = conn.execute(f"""
            SELECT DISTINCT week_number, season, MIN(report_date), MAX(report_date)
            FROM {FACT_DAILY_REPORTS}
            WHERE week_number < 1 OR week_number > 26
               OR season != {_CURRENT_SEASON}
            GROUP BY week_number, season
            ORDER BY season, week_number
        """).fetchall()

        if not bad:
            logger.info("No invalid reports found — nothing to clean up.")
            return

        for wk, season, mn, mx in bad:
            logger.info("  Bad: Week %d, season %d (%s to %s)", wk, season, mn, mx)

        result = conn.execute(f"""
            DELETE FROM {FACT_DAILY_REPORTS}
            WHERE week_number < 1 OR week_number > 26
               OR season != {_CURRENT_SEASON}
        """)
        deleted = result.fetchone()[0]
        logger.info("Deleted %d rows.", deleted)


if __name__ == "__main__":
    cleanup_bad_weeks()
