"""
debug_projections.py

Diagnostic script to trace the EXACT projection calculation for my team.
"""

from __future__ import annotations

import datetime
import logging

from src.analysis.matchup_analyzer import project_week_totals
from src.config import load_league_settings
from src.db.connection import managed_connection
from src.db.schema import (
    DIM_PLAYERS,
    FACT_DAILY_REPORTS,
    FACT_PLAYER_STATS_DAILY,
    FACT_PROJECTIONS,
    FACT_ROSTERS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def debug() -> None:
    settings = load_league_settings()
    team_key = settings.my_team_key
    today = datetime.date(2026, 4, 9)
    week = 3
    week_start = today - datetime.timedelta(days=today.weekday())  # Monday
    week_end = week_start + datetime.timedelta(days=6)
    days_remaining = max(0, (week_end - today).days)

    print(f"\ntoday={today}, week_start={week_start}, week_end={week_end}")
    print(f"days_remaining={days_remaining}, team_key={team_key}")

    with managed_connection() as conn:
        # 1. Get my roster
        roster_df = conn.execute(
            f"""
            SELECT r.player_id, p.full_name
            FROM {FACT_ROSTERS} r
            LEFT JOIN {DIM_PLAYERS} p ON r.player_id = p.player_id
            WHERE r.team_id = ? AND r.snapshot_date = ?
        """,
            [team_key, today],
        ).fetchdf()
        print(f"\n=== MY ROSTER: {len(roster_df)} players ===")
        for _, r in roster_df.iterrows():
            print(f"  {r['player_id']}: {r['full_name']}")

        if roster_df.empty:
            print("ERROR: Empty roster!")
            return

        player_ids = roster_df["player_id"].tolist()
        placeholders = ", ".join(["?" for _ in player_ids])

        # 2. Get week stats for my roster
        stats_df = conn.execute(
            f"""
            SELECT player_id,
                SUM(COALESCE(ab, 0)) AS ab,
                SUM(COALESCE(h, 0)) AS h,
                SUM(COALESCE(hr, 0)) AS hr,
                SUM(COALESCE(sb, 0)) AS sb,
                SUM(COALESCE(bb, 0)) AS bb,
                SUM(COALESCE(hbp, 0)) AS hbp,
                SUM(COALESCE(sf, 0)) AS sf,
                SUM(COALESCE(tb, 0)) AS tb,
                SUM(COALESCE(errors, 0)) AS errors,
                SUM(COALESCE(chances, 0)) AS chances,
                SUM(COALESCE(ip, 0)) AS ip,
                SUM(COALESCE(w, 0)) AS w,
                SUM(COALESCE(k, 0)) AS k,
                SUM(COALESCE(walks_allowed, 0)) AS walks_allowed,
                SUM(COALESCE(hits_allowed, 0)) AS hits_allowed,
                SUM(COALESCE(sv, 0)) AS sv,
                SUM(COALESCE(holds, 0)) AS holds
            FROM {FACT_PLAYER_STATS_DAILY}
            WHERE player_id IN ({placeholders})
              AND stat_date >= ? AND stat_date < ?
            GROUP BY player_id
        """,
            player_ids + [week_start, today],
        ).fetchdf()
        print(f"\n=== WEEK STATS: {len(stats_df)} players with stats ===")
        team_h = stats_df["h"].sum() if not stats_df.empty else 0
        team_hr = stats_df["hr"].sum() if not stats_df.empty else 0
        print(f"  Team H: {team_h}, Team HR: {team_hr}")

        # 3. Get projections (with dedup)
        proj_df = conn.execute(
            f"""
            SELECT *
            FROM {FACT_PROJECTIONS}
            WHERE player_id IN ({placeholders})
              AND target_week = ?
              AND projection_date = (
                  SELECT MAX(fp2.projection_date)
                  FROM {FACT_PROJECTIONS} fp2
                  WHERE fp2.player_id = {FACT_PROJECTIONS}.player_id
                    AND fp2.target_week = {FACT_PROJECTIONS}.target_week
              )
            ORDER BY player_id
        """,
            player_ids + [week],
        ).fetchdf()
        print(f"\n=== PROJECTIONS (deduped): {len(proj_df)} rows ===")
        if not proj_df.empty:
            print(f"  Projection dates: {proj_df['projection_date'].unique()}")
            print(f"  Sum proj_h: {proj_df['proj_h'].sum():.2f}")
            print(f"  Sum proj_hr: {proj_df['proj_hr'].sum():.2f}")
            print(f"  Sum proj_ab: {proj_df['proj_ab'].sum():.2f}")
            print(f"  Sum proj_k: {proj_df['proj_k'].sum():.2f}")

            # Show top proj_h players
            print("\n  Top proj_h (per-game rates):")
            top_h = proj_df.nlargest(5, "proj_h")[
                ["player_id", "proj_h", "proj_hr", "proj_ab", "proj_ip"]
            ]
            for _, r in top_h.iterrows():
                name = roster_df[roster_df["player_id"] == r["player_id"]][
                    "full_name"
                ].values
                name = name[0] if len(name) > 0 else "?"
                print(
                    f"    {name}: proj_h={r['proj_h']:.2f}, proj_ab={r['proj_ab']:.2f}, proj_ip={r['proj_ip']}"
                )

        # 4. Run project_week_totals
        print(f"\n=== project_week_totals(days_remaining={days_remaining}) ===")
        totals = project_week_totals(stats_df, proj_df, days_remaining)
        print(f"  Rows: {len(totals)}")
        if not totals.empty:
            print(f"  Total H: {totals['h'].sum():.2f}")
            print(f"  Total HR: {totals['hr'].sum():.2f}")
            print(f"  Total AB: {totals['ab'].sum():.2f}")
            print(f"  Total K: {totals['k'].sum():.2f}")

            # Show top H contributors
            print("\n  Top H contributors:")
            totals_with_name = totals.merge(
                roster_df[["player_id", "full_name"]], on="player_id", how="left"
            )
            top_contributors = totals_with_name.nlargest(5, "h")
            for _, r in top_contributors.iterrows():
                print(
                    f"    {r.get('full_name', '?')}: h={r['h']:.2f}, hr={r['hr']:.2f}, ab={r['ab']:.2f}"
                )

        # 5. Check WITHOUT dedup (old query behavior)
        print("\n=== WITHOUT DEDUP (old query) ===")
        proj_no_dedup = conn.execute(
            f"""
            SELECT *
            FROM {FACT_PROJECTIONS}
            WHERE player_id IN ({placeholders})
              AND target_week = ?
            ORDER BY projection_date DESC
        """,
            player_ids + [week],
        ).fetchdf()
        print(f"  Rows: {len(proj_no_dedup)} (vs {len(proj_df)} deduped)")
        if not proj_no_dedup.empty:
            totals_no_dedup = project_week_totals(
                stats_df, proj_no_dedup, days_remaining
            )
            print(f"  Total H (no dedup): {totals_no_dedup['h'].sum():.2f}")
            print(f"  Total HR (no dedup): {totals_no_dedup['hr'].sum():.2f}")

        # 6. Week 12 data
        print("\n=== WEEK 12 DATA ===")
        w12 = conn.execute(f"""
            SELECT report_date, season, week_number
            FROM {FACT_DAILY_REPORTS}
            WHERE week_number = 12
        """).fetchall()
        for rd, s, wk in w12:
            print(f"  date={rd}, season={s}, week={wk}")


if __name__ == "__main__":
    debug()
