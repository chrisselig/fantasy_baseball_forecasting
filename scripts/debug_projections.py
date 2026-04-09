"""
debug_projections.py

Diagnostic script to check projection data in MotherDuck.
Prints key metrics to diagnose why projections are inflated.
"""

from __future__ import annotations

import datetime
import json
import logging

from src.db.connection import managed_connection
from src.db.schema import (
    FACT_DAILY_REPORTS,
    FACT_PROJECTIONS,
    FACT_ROSTERS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def debug() -> None:
    with managed_connection() as conn:
        # 1. Check latest report's matchup_summary
        print("\n=== LATEST REPORT ===")
        result = conn.execute(f"""
            SELECT report_date, week_number, report_json
            FROM {FACT_DAILY_REPORTS}
            ORDER BY report_date DESC
            LIMIT 1
        """).fetchone()
        if result:
            report_date, week, report_json = result
            report = json.loads(report_json)
            print(f"Report date: {report_date}, Week: {week}")
            matchup = report.get("matchup_summary", [])
            print(f"Matchup categories: {len(matchup)}")
            for cat in matchup[:3]:
                print(f"  {cat['category']}: my={cat['my_value']}, opp={cat['opp_value']}")
            ip_pace = report.get("ip_pace", {})
            print(f"IP pace: {ip_pace}")
        else:
            print("No reports found!")

        # 2. Check projection row counts
        print("\n=== PROJECTION COUNTS ===")
        rows = conn.execute(f"""
            SELECT target_week, projection_date, COUNT(*) as cnt
            FROM {FACT_PROJECTIONS}
            GROUP BY target_week, projection_date
            ORDER BY target_week, projection_date DESC
        """).fetchall()
        for tw, pd_, cnt in rows:
            print(f"  week={tw}, date={pd_}, rows={cnt}")

        # 3. Check sample projection values (are they per-game or season totals?)
        print("\n=== SAMPLE PROJECTIONS (latest date, week 3) ===")
        samples = conn.execute(f"""
            SELECT player_id, projection_date, games_remaining, source,
                   proj_h, proj_hr, proj_ab, proj_ip, proj_k, proj_bb
            FROM {FACT_PROJECTIONS}
            WHERE target_week = 3
            ORDER BY projection_date DESC
            LIMIT 10
        """).fetchall()
        for row in samples:
            pid, pd_, gr, src, h, hr, ab, ip, k, bb = row
            print(f"  {pid}: date={pd_}, games_rem={gr}, src={src}")
            print(f"    proj_h={h}, proj_hr={hr}, proj_ab={ab}, proj_ip={ip}, proj_k={k}, proj_bb={bb}")

        # 4. Check if there are duplicate projection_dates per player
        print("\n=== DUPLICATE CHECK (week 3) ===")
        dupes = conn.execute(f"""
            SELECT player_id, COUNT(DISTINCT projection_date) as n_dates
            FROM {FACT_PROJECTIONS}
            WHERE target_week = 3
            GROUP BY player_id
            HAVING COUNT(DISTINCT projection_date) > 1
            ORDER BY n_dates DESC
            LIMIT 5
        """).fetchall()
        if dupes:
            for pid, n in dupes:
                print(f"  {pid}: {n} projection dates")
        else:
            print("  No duplicates — all players have 1 projection date")

        # 5. Check what dedup query returns
        print("\n=== DEDUP QUERY ROW COUNT (week 3) ===")
        dedup_count = conn.execute(f"""
            SELECT COUNT(*)
            FROM {FACT_PROJECTIONS}
            WHERE target_week = 3
              AND projection_date = (
                  SELECT MAX(fp2.projection_date)
                  FROM {FACT_PROJECTIONS} fp2
                  WHERE fp2.player_id = {FACT_PROJECTIONS}.player_id
                    AND fp2.target_week = {FACT_PROJECTIONS}.target_week
              )
        """).fetchone()
        total_count = conn.execute(f"""
            SELECT COUNT(*)
            FROM {FACT_PROJECTIONS}
            WHERE target_week = 3
        """).fetchone()
        print(f"  Dedup count: {dedup_count[0]}, Total count: {total_count[0]}")

        # 6. Check roster size
        print("\n=== ROSTER SIZE ===")
        roster_info = conn.execute(f"""
            SELECT snapshot_date, team_id, COUNT(*) as n_players
            FROM {FACT_ROSTERS}
            GROUP BY snapshot_date, team_id
            ORDER BY snapshot_date DESC
            LIMIT 10
        """).fetchall()
        for sd, tid, n in roster_info:
            print(f"  date={sd}, team={tid}, players={n}")

        # 7. Check weeks in daily reports
        print("\n=== WEEKS IN DAILY REPORTS ===")
        weeks = conn.execute(f"""
            SELECT DISTINCT week_number, COUNT(*) as n_reports,
                   MIN(report_date) as min_date, MAX(report_date) as max_date
            FROM {FACT_DAILY_REPORTS}
            GROUP BY week_number
            ORDER BY week_number
        """).fetchall()
        for wk, n, mn, mx in weeks:
            print(f"  Week {wk}: {n} reports ({mn} to {mx})")


if __name__ == "__main__":
    debug()
