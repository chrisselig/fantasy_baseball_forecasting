#!/usr/bin/env python3
"""scripts/seed_motherduck_2025.py

Populate MotherDuck with real 2025 MLB stats and run the daily pipeline.

Fetches actual 2025 season stats via pybaseball, randomly assigns players
from the known pool to two fantasy teams, writes everything to MotherDuck,
and runs the full pipeline so the deployed Shiny app shows real 2025 data.

Usage::

    MOTHERDUCK_TOKEN=<token> python scripts/seed_motherduck_2025.py [--seed N] [--dry-run] [--verbose]

The --dry-run flag writes to an in-memory DB instead of MotherDuck so you
can verify the script before touching production data.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import logging
import random
import sys
import warnings
from pathlib import Path
from unittest import mock

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_league_settings
from src.db.connection import managed_connection
from src.db.loaders_mlb import get_fantasy_week
from src.db.schema import (
    DIM_PLAYERS,
    FACT_MATCHUPS,
    FACT_PLAYER_STATS_DAILY,
    FACT_ROSTERS,
    create_all_tables,
)
from src.pipeline.daily_run import run_daily_pipeline

logger = logging.getLogger(__name__)

# ── Run parameters ────────────────────────────────────────────────────────────

# Mid-late 2025 season window: week 23 of a typical fantasy calendar.
# RUN_DATE is a Wednesday so the week has started and there are stats to show.
RUN_DATE = datetime.date(2025, 9, 17)  # Wednesday
# Pipeline query: stat_date >= week_start AND stat_date < run_date.
# Week starts Monday 2025-09-15, so a stat_date of 2025-09-15 is included.
STAT_DATE = datetime.date(2025, 9, 15)  # Monday of run week

MY_TEAM_KEY = "422.l.87941.t.3"
OPP_TEAM_KEY = "422.l.87941.t.5"

_LEAGUE_ID = 87941
_SEASON = 2025
# Compute week using the same formula as the pipeline so matchup lookup succeeds.
_SEASON_START_2025 = datetime.date(2025, 3, 27)
_WEEK = get_fantasy_week(RUN_DATE, _SEASON_START_2025)

# ── Player pool ───────────────────────────────────────────────────────────────
# (full_name, mlb_id, fg_id, team, eligible_positions, bats, throws)
# fg_id must match the IDfg column returned by pybaseball.batting/pitching_stats.

_PLAYER_POOL: list[tuple] = [
    # ── Catchers ──────────────────────────────────────────────────────────────
    ("William Contreras", 661388, "24204", "MIL", ["C"], "R", "R"),
    ("Adley Rutschman", 668939, "26416", "BAL", ["C"], "S", "R"),
    ("Salvador Perez", 521692, "14295", "KC", ["C"], "R", "R"),
    ("Sean Murphy", 663738, "22756", "ATL", ["C"], "R", "R"),
    # ── 1B ────────────────────────────────────────────────────────────────────
    ("Freddie Freeman", 518692, "13853", "LAD", ["1B"], "L", "R"),
    ("Pete Alonso", 624413, "19611", "NYM", ["1B"], "R", "R"),
    ("Christian Walker", 572233, "8167", "HOU", ["1B"], "R", "R"),
    ("Matt Olson", 621566, "15021", "ATL", ["1B"], "L", "R"),
    # ── 2B ────────────────────────────────────────────────────────────────────
    ("Jose Altuve", 514888, "7579", "HOU", ["2B"], "R", "R"),
    ("Ozzie Albies", 645277, "18401", "ATL", ["2B"], "S", "R"),
    ("Gleyber Torres", 650402, "21290", "NYY", ["2B", "SS"], "R", "R"),
    ("Jeff McNeil", 643446, "17461", "NYM", ["2B", "OF"], "L", "R"),
    # ── 3B ────────────────────────────────────────────────────────────────────
    ("Jose Ramirez", 608070, "13510", "CLE", ["3B"], "S", "R"),
    ("Austin Riley", 663586, "19849", "ATL", ["3B"], "R", "R"),
    ("Rafael Devers", 646240, "20123", "BOS", ["3B"], "L", "R"),
    ("Manny Machado", 592518, "11579", "SD", ["3B", "SS"], "R", "R"),
    # ── SS ────────────────────────────────────────────────────────────────────
    ("Bobby Witt Jr", 677951, "22468", "KC", ["SS"], "R", "R"),
    ("Corey Seager", 608369, "13624", "TEX", ["SS"], "L", "R"),
    ("Francisco Lindor", 596019, "12916", "NYM", ["SS"], "S", "R"),
    ("Gunnar Henderson", 683002, "22563", "BAL", ["SS", "3B"], "L", "R"),
    ("CJ Abrams", 682928, "23698", "WSH", ["SS"], "L", "R"),
    # ── OF ────────────────────────────────────────────────────────────────────
    ("Aaron Judge", 592450, "15640", "NYY", ["OF"], "R", "R"),
    ("Ronald Acuna Jr", 660670, "20785", "ATL", ["OF"], "R", "R"),
    ("Kyle Tucker", 663656, "20655", "HOU", ["OF"], "L", "R"),
    ("Juan Soto", 665742, "20596", "NYM", ["OF"], "L", "L"),
    ("Yordan Alvarez", 670541, "20560", "HOU", ["OF"], "L", "R"),
    ("Julio Rodriguez", 677594, "22462", "SEA", ["OF"], "R", "R"),
    ("Jackson Chourio", 694192, "25764", "MIL", ["OF"], "S", "R"),
    ("Fernando Tatis Jr", 665487, "20451", "SD", ["SS", "OF"], "R", "R"),
    ("Michael Harris II", 682998, "22562", "ATL", ["OF"], "L", "L"),
    ("Luis Robert Jr", 673357, "20757", "CWS", ["OF"], "R", "R"),
    ("Cedric Mullins", 656775, "21074", "BAL", ["OF"], "S", "L"),
    ("Jorge Soler", 596747, "15406", "MIA", ["OF"], "R", "R"),
    ("Steven Kwan", 680757, "21710", "CLE", ["OF"], "L", "L"),
    ("Teoscar Hernandez", 606466, "16624", "LAD", ["OF"], "R", "R"),
    ("Jarren Duran", 680776, "22621", "BOS", ["OF"], "L", "L"),
    ("Marcell Ozuna", 542303, "11477", "ATL", ["OF"], "R", "R"),
    ("Wyatt Langford", 691234, "25765", "TEX", ["OF"], "R", "R"),
    ("Randy Arozarena", 668227, "22027", "SEA", ["OF"], "R", "R"),
    ("Anthony Santander", 606061, "16164", "TOR", ["OF"], "S", "R"),
    # ── SP ────────────────────────────────────────────────────────────────────
    ("Zack Wheeler", 554430, "12716", "PHI", ["SP"], "R", "R"),
    ("Corbin Burnes", 669203, "21538", "BAL", ["SP"], "R", "R"),
    ("Gerrit Cole", 543037, "10155", "NYY", ["SP"], "R", "R"),
    ("Dylan Cease", 656302, "21163", "SD", ["SP"], "R", "R"),
    ("Aaron Nola", 605400, "15594", "PHI", ["SP"], "R", "R"),
    ("Kevin Gausman", 592332, "14798", "TOR", ["SP"], "R", "R"),
    ("Logan Webb", 657277, "21461", "SF", ["SP"], "R", "R"),
    ("Pablo Lopez", 641154, "20663", "MIN", ["SP"], "R", "R"),
    ("Yoshinobu Yamamoto", 808963, "sa3011279", "LAD", ["SP"], "R", "R"),
    ("Max Fried", 594789, "13109", "NYY", ["SP"], "L", "L"),
    # ── RP ────────────────────────────────────────────────────────────────────
    ("Josh Hader", 624578, "17626", "HOU", ["RP"], "L", "L"),
    ("Emmanuel Clase", 667555, "21463", "CLE", ["RP"], "R", "R"),
    ("Ryan Helsley", 651529, "21090", "STL", ["RP"], "R", "R"),
    ("David Bednar", 676969, "22268", "PIT", ["RP"], "R", "R"),
    ("Alexis Diaz", 672515, "26424", "CIN", ["RP"], "R", "R"),
    ("Jordan Romano", 669270, "25916", "TOR", ["RP"], "R", "R"),
    ("Pete Fairbanks", 656432, "22007", "TB", ["RP"], "R", "R"),
    ("Camilo Doval", 672717, "22767", "SF", ["RP"], "R", "R"),
]


# ── Roster-slot assignment (mirrors seed_2025_test.py) ────────────────────────

_ACTIVE_SLOTS = [
    "C",
    "1B",
    "2B",
    "3B",
    "SS",
    "OF",
    "OF",
    "OF",
    "Util",
    "Util",
    "SP",
    "SP",
    "RP",
    "RP",
    "P",
    "P",
]
_BENCH_SLOTS = ["BN"] * 4


def _can_fill(slot: str, player: tuple) -> bool:
    positions = player[4]
    if slot in ("C", "1B", "2B", "3B", "SS", "OF"):
        return slot in positions
    if slot == "Util":
        return not all(p in ("SP", "RP") for p in positions)
    if slot == "SP":
        return "SP" in positions
    if slot == "RP":
        return "RP" in positions
    if slot == "P":
        return "SP" in positions or "RP" in positions
    return True  # BN


def _assign_roster(pool: list[tuple], slots: list[str]) -> list[tuple[str, tuple]]:
    remaining = list(pool)
    assigned: list[tuple[str, tuple]] = []
    for slot in slots:
        for i, p in enumerate(remaining):
            if _can_fill(slot, p):
                assigned.append((slot, p))
                remaining.pop(i)
                break
    return assigned


def build_rosters(rng: random.Random) -> tuple[list, list]:
    """Randomly pick and assign two full teams from the player pool."""
    catchers = [p for p in _PLAYER_POOL if "C" in p[4]]
    first_base = [p for p in _PLAYER_POOL if "1B" in p[4]]
    second_base = [p for p in _PLAYER_POOL if "2B" in p[4]]
    third_base = [p for p in _PLAYER_POOL if "3B" in p[4]]
    shortstop = [p for p in _PLAYER_POOL if "SS" in p[4]]
    outfield = [p for p in _PLAYER_POOL if "OF" in p[4]]
    starters = [p for p in _PLAYER_POOL if "SP" in p[4]]
    relievers = [p for p in _PLAYER_POOL if "RP" in p[4]]

    used: set[int] = set()

    def pick_unique(pool: list, n: int) -> list:
        available = [p for p in pool if p[1] not in used]
        rng.shuffle(available)
        chosen = available[:n]
        for p in chosen:
            used.add(p[1])
        return chosen

    def build_pool() -> list[tuple]:
        team: list[tuple] = []
        team += pick_unique(catchers, 1)
        team += pick_unique(first_base, 1)
        team += pick_unique(second_base, 1)
        team += pick_unique(third_base, 1)
        team += pick_unique(shortstop, 1)
        team += pick_unique(outfield, 5)
        team += pick_unique(starters, 3)
        team += pick_unique(relievers, 3)
        leftovers = (
            [p for p in outfield if p[1] not in used]
            + [p for p in first_base if p[1] not in used]
            + [p for p in second_base if p[1] not in used]
        )
        team += pick_unique(leftovers, 4)
        return team

    all_slots = _ACTIVE_SLOTS + _BENCH_SLOTS
    my_pool = build_pool()
    opp_pool = build_pool()
    my_roster = _assign_roster(my_pool, all_slots)
    opp_roster = _assign_roster(opp_pool, all_slots)
    return my_roster, opp_roster


# ── Pybaseball data fetch ─────────────────────────────────────────────────────


def _fetch_real_stats(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch real batting and pitching stats for *season* via pybaseball.

    Returns:
        (bat_df, pit_df) — both keyed on 'IDfg' (str) and 'Name'.
        Columns are the raw pybaseball output; callers do their own mapping.
    """
    import pybaseball  # noqa: PLC0415

    print(f"\n  Fetching {season} batting stats from pybaseball (FanGraphs)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # qual=1 disables the PA qualifier so all players appear, not just qualified batters
        bat_df: pd.DataFrame = pybaseball.batting_stats(season, qual=1)
    print(f"  Got {len(bat_df)} batters.")

    print(f"  Fetching {season} pitching stats from pybaseball (FanGraphs)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # qual=1 disables the IP qualifier so all pitchers appear
        pit_df: pd.DataFrame = pybaseball.pitching_stats(season, qual=1)
    print(f"  Got {len(pit_df)} pitchers.")

    # Normalise IDfg to string to match our fg_id strings
    for df in (bat_df, pit_df):
        if "IDfg" in df.columns:
            df["IDfg"] = df["IDfg"].astype(str).str.strip()

    return bat_df, pit_df


def _match_player_stats(
    player: tuple,
    bat_df: pd.DataFrame,
    pit_df: pd.DataFrame,
) -> dict | None:
    """Return a stat dict for *player* from pybaseball DataFrames.

    Attempts to match by fg_id first, then by full_name.  Returns None if
    no match is found (caller falls back to zeros or skips).
    """
    name, mlb_id, fg_id, team, positions, bats, throws = player
    is_pitcher = all(p in ("SP", "RP") for p in positions)
    source_df = pit_df if is_pitcher else bat_df

    row: pd.Series | None = None

    # 1. Match by FanGraphs ID
    if fg_id and "IDfg" in source_df.columns:
        matches = source_df[source_df["IDfg"] == str(fg_id)]
        if not matches.empty:
            row = matches.iloc[0]

    # 2. Fallback: match by name
    if row is None and "Name" in source_df.columns:
        # Exact match first
        matches = source_df[source_df["Name"].str.strip() == name]
        if not matches.empty:
            row = matches.iloc[0]
        else:
            # Partial last-name match (handles "Jr" / accent differences)
            last = name.split()[-1].lower().rstrip(".")
            fuzzy = source_df[
                source_df["Name"].str.lower().str.contains(last, regex=False)
            ]
            if len(fuzzy) == 1:
                row = fuzzy.iloc[0]

    if row is None:
        logger.warning(
            "No pybaseball match for %s (fg_id=%s) — using zeros", name, fg_id
        )
        return None

    if is_pitcher:
        return _map_pitcher_row(row)
    return _map_batter_row(row)


def _map_batter_row(row: pd.Series) -> dict:
    """Map a raw pybaseball batting row to fact_player_stats_daily columns."""

    def g(col: str, default: float = 0.0) -> float:
        v = row.get(col, default)
        try:
            return float(v) if pd.notna(v) else default
        except (TypeError, ValueError):
            return default

    ab = int(g("AB"))
    h = int(g("H"))
    hr = int(g("HR"))
    sb = int(g("SB"))
    bb = int(g("BB"))
    hbp = int(g("HBP"))
    sf = int(g("SF"))

    singles = max(0, h - int(g("2B")) - int(g("3B")) - hr)
    tb = singles + 2 * int(g("2B")) + 3 * int(g("3B")) + 4 * hr

    avg = round(h / ab, 3) if ab > 0 else 0.0
    obp_den = ab + bb + hbp + sf
    obp = round((h + bb + hbp) / obp_den, 3) if obp_den > 0 else 0.0
    slg = round(tb / ab, 3) if ab > 0 else 0.0
    ops = round(obp + slg, 3)

    return {
        "ab": ab,
        "h": h,
        "hr": hr,
        "sb": sb,
        "bb": bb,
        "hbp": hbp,
        "sf": sf,
        "tb": tb,
        "avg": avg,
        "ops": ops,
        "fpct": None,
        "errors": None,
        "chances": None,
        "ip": None,
        "w": None,
        "k": None,
        "walks_allowed": None,
        "hits_allowed": None,
        "sv": None,
        "holds": None,
        "whip": None,
        "k_bb": None,
        "sv_h": None,
    }


def _map_pitcher_row(row: pd.Series) -> dict:
    """Map a raw pybaseball pitching row to fact_player_stats_daily columns."""

    def g(col: str, default: float = 0.0) -> float:
        v = row.get(col, default)
        try:
            return float(v) if pd.notna(v) else default
        except (TypeError, ValueError):
            return default

    ip = round(g("IP"), 1)
    w = int(g("W"))
    k = int(g("SO"))
    bb = int(g("BB"))
    ha = int(g("H"))
    sv = int(g("SV"))
    hld = int(g("HLD"))
    sv_h = sv + hld
    whip = round((bb + ha) / ip, 3) if ip > 0 else 0.0
    k_bb = round(k / bb, 2) if bb > 0 else float(k)

    return {
        "ip": ip,
        "w": w,
        "k": k,
        "walks_allowed": bb,
        "hits_allowed": ha,
        "sv": sv,
        "holds": hld,
        "sv_h": sv_h,
        "whip": whip,
        "k_bb": k_bb,
        "ab": None,
        "h": None,
        "hr": None,
        "sb": None,
        "bb": None,
        "hbp": None,
        "sf": None,
        "tb": None,
        "errors": None,
        "chances": None,
        "avg": None,
        "ops": None,
        "fpct": None,
    }


# ── DB population ─────────────────────────────────────────────────────────────


def populate_db(
    conn: duckdb.DuckDBPyConnection,
    my_roster: list,
    opp_roster: list,
    bat_df: pd.DataFrame,
    pit_df: pd.DataFrame,
) -> None:
    """Create tables (if needed) and insert roster + stats data."""
    create_all_tables(conn)

    # Deduplicate players across both rosters
    seen: set[int] = set()
    players: list[tuple] = []
    for _, p in my_roster + opp_roster:
        if p[1] not in seen:
            players.append(p)
            seen.add(p[1])

    # ── dim_players ────────────────────────────────────────────────────────────
    dim_rows = [
        {
            "player_id": f"422.p.{mlb_id}",
            "mlb_id": mlb_id,
            "fg_id": fg_id,
            "full_name": name,
            "team": team,
            "positions": positions,
            "bats": bats,
            "throws": throws,
            "status": "Active",
            "updated_at": datetime.datetime.now(),
        }
        for name, mlb_id, fg_id, team, positions, bats, throws in players
    ]
    dim_df = pd.DataFrame(dim_rows)
    conn.register("_dim_tmp", dim_df)
    conn.execute(f"""
        INSERT OR REPLACE INTO {DIM_PLAYERS}
            (player_id, mlb_id, fg_id, full_name, team, positions,
             bats, throws, status, updated_at)
        SELECT player_id, mlb_id, fg_id, full_name, team, positions,
               bats, throws, status, updated_at
        FROM _dim_tmp
    """)
    conn.unregister("_dim_tmp")
    print(f"  Inserted {len(dim_rows)} players into {DIM_PLAYERS}.")

    # ── fact_rosters ───────────────────────────────────────────────────────────
    roster_rows = [
        {
            "team_id": team_key,
            "player_id": f"422.p.{player[1]}",
            "snapshot_date": RUN_DATE,
            "roster_slot": slot,
            "acquisition_type": "draft",
        }
        for team_key, assignments in [
            (MY_TEAM_KEY, my_roster),
            (OPP_TEAM_KEY, opp_roster),
        ]
        for slot, player in assignments
    ]
    roster_df = pd.DataFrame(roster_rows)
    conn.register("_roster_tmp", roster_df)
    conn.execute(f"""
        INSERT OR REPLACE INTO {FACT_ROSTERS}
            (team_id, player_id, snapshot_date, roster_slot, acquisition_type)
        SELECT team_id, player_id, snapshot_date, roster_slot, acquisition_type
        FROM _roster_tmp
    """)
    conn.unregister("_roster_tmp")
    print(f"  Inserted {len(roster_rows)} roster rows into {FACT_ROSTERS}.")

    # ── fact_player_stats_daily ────────────────────────────────────────────────
    # Write a single row per player tagged to STAT_DATE (season totals).
    stat_rows = []
    matched = 0
    unmatched_names = []
    for player in players:
        name, mlb_id, *_ = player
        player_id = f"422.p.{mlb_id}"
        stats = _match_player_stats(player, bat_df, pit_df)
        if stats is None:
            unmatched_names.append(name)
            # Write zeros so the pipeline can still run
            positions = player[4]
            is_pitcher = all(p in ("SP", "RP") for p in positions)
            if is_pitcher:
                stats = {
                    "ip": 0.0,
                    "w": 0,
                    "k": 0,
                    "walks_allowed": 0,
                    "hits_allowed": 0,
                    "sv": 0,
                    "holds": 0,
                    "sv_h": 0,
                    "whip": 0.0,
                    "k_bb": 0.0,
                    "ab": None,
                    "h": None,
                    "hr": None,
                    "sb": None,
                    "bb": None,
                    "hbp": None,
                    "sf": None,
                    "tb": None,
                    "errors": None,
                    "chances": None,
                    "avg": None,
                    "ops": None,
                    "fpct": None,
                }
            else:
                stats = {
                    "ab": 0,
                    "h": 0,
                    "hr": 0,
                    "sb": 0,
                    "bb": 0,
                    "hbp": 0,
                    "sf": 0,
                    "tb": 0,
                    "avg": 0.0,
                    "ops": 0.0,
                    "fpct": None,
                    "errors": None,
                    "chances": None,
                    "ip": None,
                    "w": None,
                    "k": None,
                    "walks_allowed": None,
                    "hits_allowed": None,
                    "sv": None,
                    "holds": None,
                    "whip": None,
                    "k_bb": None,
                    "sv_h": None,
                }
        else:
            matched += 1

        stats["player_id"] = player_id
        stats["stat_date"] = STAT_DATE
        stat_rows.append(stats)

    if unmatched_names:
        print(
            f"  Warning: {len(unmatched_names)} players not matched in pybaseball data:"
        )
        for n in unmatched_names:
            print(f"    - {n}")

    stat_df = pd.DataFrame(stat_rows)
    conn.register("_stat_tmp", stat_df)
    conn.execute(f"""
        INSERT OR REPLACE INTO {FACT_PLAYER_STATS_DAILY}
            (player_id, stat_date, ab, h, hr, sb, bb, hbp, sf, tb,
             errors, chances, ip, w, k, walks_allowed, hits_allowed,
             sv, holds, avg, ops, fpct, whip, k_bb, sv_h)
        SELECT player_id, stat_date, ab, h, hr, sb, bb, hbp, sf, tb,
               errors, chances, ip, w, k, walks_allowed, hits_allowed,
               sv, holds, avg, ops, fpct, whip, k_bb, sv_h
        FROM _stat_tmp
    """)
    conn.unregister("_stat_tmp")
    print(
        f"  Inserted {len(stat_rows)} stat rows ({matched} with real data, "
        f"{len(unmatched_names)} zeroed) into {FACT_PLAYER_STATS_DAILY}."
    )

    # ── fact_matchups ──────────────────────────────────────────────────────────
    matchup_id = f"{_LEAGUE_ID}_{_SEASON}_W{_WEEK:02d}_T3vsT5"
    conn.execute(
        f"""
        INSERT OR REPLACE INTO {FACT_MATCHUPS}
            (matchup_id, league_id, week_number, season, team_id_home, team_id_away)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        [matchup_id, _LEAGUE_ID, _WEEK, _SEASON, MY_TEAM_KEY, OPP_TEAM_KEY],
    )
    print(f"  Inserted matchup {matchup_id} into {FACT_MATCHUPS}.")


# ── Report printer ────────────────────────────────────────────────────────────


def print_report(report: dict, conn: duckdb.DuckDBPyConnection) -> None:
    print(f"\n{'═' * 62}")
    print(
        f"  DAILY REPORT — {report.get('report_date', '?')}  (Week {report.get('week_number', '?')})"
    )
    print(f"{'═' * 62}")

    matchup = report.get("matchup_summary", [])
    if matchup:
        wins = sum(1 for r in matchup if r.get("my_leads"))
        losses = sum(
            1 for r in matchup if not r.get("my_leads") and r.get("status") != "toss_up"
        )
        ties = sum(1 for r in matchup if r.get("status") == "toss_up")
        print("\n  Matchup vs Opponent (projected)")
        print(f"  {'Category':<10}  {'Mine':>10}  {'Opp':>10}  Status")
        print(f"  {'─' * 50}")
        for row in matchup:
            cat = row.get("category", "?").upper()
            mine = row.get("my_value", 0)
            opp = row.get("opp_value", 0)
            status = row.get("status", "?")
            sym = (
                "✓"
                if row.get("my_leads")
                else ("✗" if status not in ("toss_up",) else "~")
            )
            # Format floats nicely
            try:
                mine_s = f"{float(mine):.3f}"
                opp_s = f"{float(opp):.3f}"
            except (TypeError, ValueError):
                mine_s = str(mine)
                opp_s = str(opp)
            print(f"  {sym} {cat:<9}  {mine_s:>10}  {opp_s:>10}  {status}")
        print(f"\n  Projected: leading {wins}, trailing {losses}, tied {ties}")

    lineup = report.get("lineup", {})
    if lineup:
        pid_list = list(lineup.values())
        placeholders = ", ".join("?" * len(pid_list))
        name_map = (
            dict(
                conn.execute(
                    f"SELECT player_id, full_name FROM {DIM_PLAYERS} WHERE player_id IN ({placeholders})",
                    pid_list,
                ).fetchall()
            )
            if pid_list
            else {}
        )
        print("\n  Recommended Lineup")
        print(f"  {'─' * 40}")
        for slot, pid in lineup.items():
            print(f"  {slot:<8}  {name_map.get(pid, pid)}")

    adds = report.get("adds", [])
    if adds:
        print(f"\n  Waiver Wire Recommendations ({len(adds)}):")
        for a in adds[:5]:
            print(
                f"  +  {a.get('add_player_id', '?')}  (drop {a.get('drop_player_id', '?')})"
            )
            print(f"       reason: {a.get('reason', '?')}")

    ip = report.get("ip_pace", {})
    if ip:
        cur = float(ip.get("current_ip", 0))
        proj = float(ip.get("projected_ip", 0))
        need = int(ip.get("min_ip", 21))
        sym = "✓" if ip.get("on_pace") else "⚠ "
        print(
            f"\n  IP Pace: {cur:.1f} actual, {proj:.1f} projected, need {need} — {sym}"
        )

    print()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed MotherDuck with real 2025 MLB stats and run the pipeline"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write to in-memory DuckDB instead of MotherDuck",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    parser.add_argument("--json", action="store_true", help="Dump raw report JSON")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    rng = random.Random(args.seed)

    print(f"\n{'═' * 62}")
    print("  Fantasy Baseball 2025 — MotherDuck Seeder")
    print(f"  Run date  : {RUN_DATE}  (fantasy week {_WEEK}, 2025 season)")
    print(f"  Stat date : {STAT_DATE}  (full-season totals tagged to this date)")
    print(f"  Seed      : {args.seed}")
    if args.dry_run:
        print("  Mode      : DRY RUN (in-memory DuckDB — MotherDuck NOT modified)")
    else:
        print("  Mode      : LIVE (writing to MotherDuck)")
    print(f"{'═' * 62}")

    # Fetch real 2025 stats
    bat_df, pit_df = _fetch_real_stats(_SEASON)

    # Assign rosters
    my_roster, opp_roster = build_rosters(rng)
    print(f"\n  My team  : {len(my_roster)} slots  ({MY_TEAM_KEY})")
    for slot, (name, _, _, team, pos, _, _) in my_roster:
        print(f"    {slot:<8}  {name:<24}  {'/'.join(pos):<10}  {team}")
    print(f"\n  Opponent : {len(opp_roster)} slots  ({OPP_TEAM_KEY})")
    for slot, (name, _, _, team, pos, _, _) in opp_roster:
        print(f"    {slot:<8}  {name:<24}  {'/'.join(pos):<10}  {team}")

    # Open connection and populate
    if args.dry_run:
        conn = duckdb.connect(":memory:")
        try:
            print("\n  Populating in-memory DuckDB...")
            populate_db(conn, my_roster, opp_roster, bat_df, pit_df)
            _run_pipeline(conn, my_roster, args.json)
        finally:
            conn.close()
    else:
        with managed_connection() as conn:
            print("\n  Populating MotherDuck...")
            populate_db(conn, my_roster, opp_roster, bat_df, pit_df)
            _run_pipeline(conn, my_roster, args.json)


def _run_pipeline(
    conn: duckdb.DuckDBPyConnection,
    my_roster: list,
    dump_json: bool,
) -> None:
    """Run the daily pipeline and print the resulting report."""
    settings = dataclasses.replace(load_league_settings(), my_team_key=MY_TEAM_KEY)

    # Give all players a "game today" — pass player_ids as the schedule mock
    all_pids = conn.execute(f"SELECT player_id FROM {DIM_PLAYERS}").fetchdf()[
        ["player_id"]
    ]

    with (
        mock.patch(
            "src.pipeline.daily_run._step_load_yahoo",
            return_value=({}, pd.DataFrame()),
        ),
        mock.patch(
            "src.pipeline.daily_run._step_load_mlb_stats",
            return_value=({}, all_pids),
        ),
        mock.patch(
            "src.pipeline.daily_run._step_refresh_projections",
            return_value={},
        ),
    ):
        print("\n  Running daily pipeline...")
        result = run_daily_pipeline(conn, settings, run_date=RUN_DATE)

    status = result["status"]
    rows = result["rows_written"]
    print(f"  Status      : {status}")
    print(f"  Rows written: {rows}")

    row = conn.execute(
        "SELECT report_json FROM fact_daily_reports WHERE report_date = ?",
        [RUN_DATE],
    ).fetchone()

    if row and row[0]:
        report = json.loads(row[0]) if isinstance(row[0], str) else row[0]
        print_report(report, conn)
        if dump_json:
            print("\n── Raw JSON ──────────────────────────────────────────────")
            print(json.dumps(report, indent=2, default=str))
    else:
        print("\n  [No report written to DB]")


if __name__ == "__main__":
    main()
