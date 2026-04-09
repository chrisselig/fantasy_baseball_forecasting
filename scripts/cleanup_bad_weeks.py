"""
cleanup_bad_weeks.py

One-time script to remove invalid week data from fact_daily_reports.
Deletes any rows where week_number is outside the valid range (1–26).

Usage::

    python -m scripts.cleanup_bad_weeks
"""

from __future__ import annotations

import logging

from src.db.connection import managed_connection
from src.db.schema import FACT_DAILY_REPORTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def cleanup_bad_weeks() -> None:
    """Delete rows from fact_daily_reports with invalid week numbers."""
    with managed_connection() as conn:
        # Show what will be deleted
        bad = conn.execute(f"""
            SELECT DISTINCT week_number
            FROM {FACT_DAILY_REPORTS}
            WHERE week_number < 1 OR week_number > 26
            ORDER BY week_number
        """).fetchall()

        if not bad:
            logger.info("No invalid weeks found — nothing to clean up.")
            return

        bad_weeks = [r[0] for r in bad]
        logger.info("Found invalid weeks: %s — deleting...", bad_weeks)

        result = conn.execute(f"""
            DELETE FROM {FACT_DAILY_REPORTS}
            WHERE week_number < 1 OR week_number > 26
        """)
        deleted = result.fetchone()[0]
        logger.info("Deleted %d rows with invalid week numbers.", deleted)


if __name__ == "__main__":
    cleanup_bad_weeks()
