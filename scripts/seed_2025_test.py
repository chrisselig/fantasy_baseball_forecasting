#!/usr/bin/env python3
"""scripts/seed_2025_test.py

End-to-end pipeline test using 2025 MLB season data.

Randomly assigns real 2025 MLB players to "my team" and an opponent based on
league roster settings, pre-populates a local in-memory DuckDB with realistic
per-day stats, mocks the external API steps, and runs the full pipeline.

Usage::

    python scripts/seed_2025_test.py [--seed N] [--verbose] [--json]
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import logging
import random
import sys
from pathlib import Path
from unittest import mock

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_league_settings
from src.db.schema import (
    DIM_PLAYERS,
    FACT_MATCHUPS,
    FACT_PLAYER_STATS_DAILY,
    FACT_ROSTERS,
    create_all_tables,
)
from src.pipeline.daily_run import run_daily_pipeline

# ── Constants ─────────────────────────────────────────────────────────────────

# Mid-season Thursday → fantasy week 21, well within regular season.
RUN_DATE = datetime.date(2025, 8, 14)
_WEEK_START = datetime.date(2025, 8, 11)   # Monday of that week
STAT_DATES = [_WEEK_START + datetime.timedelta(days=i) for i in range(3)]  # Mon–Wed

MY_TEAM_KEY = "422.l.87941.t.3"
OPP_TEAM_KEY = "422.l.87941.t.5"

# ── Player pool ───────────────────────────────────────────────────────────────
# (full_name, mlb_id, fg_id_or_None, team, eligible_positions, bats, throws)

_PLAYER_POOL: list[tuple] = [
    # ── Catchers ──────────────────────────────────────────────────────────────
    ("William Contreras",     661388, "24204",    "MIL", ["C"],           "R", "R"),
    ("Adley Rutschman",       668939, None,        "BAL", ["C"],           "S", "R"),
    ("Salvador Perez",        521692, "14295",     "KC",  ["C"],           "R", "R"),
    ("Sean Murphy",           663738, None,        "ATL", ["C"],           "R", "R"),
    # ── 1B ────────────────────────────────────────────────────────────────────
    ("Freddie Freeman",       518692, "1109546",   "LAD", ["1B"],          "L", "R"),
    ("Pete Alonso",           624413, "19611",     "NYM", ["1B"],          "R", "R"),
    ("Christian Walker",      572233, "8167",      "HOU", ["1B"],          "R", "R"),
    ("Matt Olson",            621566, "15021",     "ATL", ["1B"],          "L", "R"),
    # ── 2B ────────────────────────────────────────────────────────────────────
    ("Jose Altuve",           514888, "7579",      "HOU", ["2B"],          "R", "R"),
    ("Ozzie Albies",          645277, "18401",     "ATL", ["2B"],          "S", "R"),
    ("Gleyber Torres",        650402, "21290",     "NYY", ["2B", "SS"],    "R", "R"),
    ("Jeff McNeil",           643446, "17461",     "NYM", ["2B", "OF"],    "L", "R"),
    # ── 3B ────────────────────────────────────────────────────────────────────
    ("Jose Ramirez",          608070, "13510",     "CLE", ["3B"],          "S", "R"),
    ("Austin Riley",          663586, "19849",     "ATL", ["3B"],          "R", "R"),
    ("Rafael Devers",         646240, "20123",     "BOS", ["3B"],          "L", "R"),
    ("Manny Machado",         592518, "11579",     "SD",  ["3B", "SS"],    "R", "R"),
    # ── SS ────────────────────────────────────────────────────────────────────
    ("Bobby Witt Jr",         677951, "22468",     "KC",  ["SS"],          "R", "R"),
    ("Corey Seager",          608369, "13624",     "TEX", ["SS"],          "L", "R"),
    ("Francisco Lindor",      596019, "12916",     "NYM", ["SS"],          "S", "R"),
    ("Gunnar Henderson",      683002, "22563",     "BAL", ["SS", "3B"],    "L", "R"),
    ("CJ Abrams",             682928, "23698",     "WSH", ["SS"],          "L", "R"),
    # ── OF ────────────────────────────────────────────────────────────────────
    ("Aaron Judge",           592450, "15640",     "NYY", ["OF"],          "R", "R"),
    ("Ronald Acuna Jr",       660670, "20785",     "ATL", ["OF"],          "R", "R"),
    ("Kyle Tucker",           663656, "20655",     "HOU", ["OF"],          "L", "R"),
    ("Juan Soto",             665742, "20596",     "NYM", ["OF"],          "L", "L"),
    ("Yordan Alvarez",        670541, "20560",     "HOU", ["OF"],          "L", "R"),
    ("Julio Rodriguez",       677594, "22462",     "SEA", ["OF"],          "R", "R"),
    ("Jackson Chourio",       694192, "25764",     "MIL", ["OF"],          "S", "R"),
    ("Fernando Tatis Jr",     665487, "20451",     "SD",  ["SS", "OF"],    "R", "R"),
    ("Michael Harris II",     682998, "22562",     "ATL", ["OF"],          "L", "L"),
    ("Luis Robert Jr",        673357, "20757",     "CWS", ["OF"],          "R", "R"),
    ("Cedric Mullins",        656775, "21074",     "BAL", ["OF"],          "S", "L"),
    ("Jorge Soler",           596747, "15406",     "MIA", ["OF"],          "R", "R"),
    ("Steven Kwan",           680757, "21710",     "CLE", ["OF"],          "L", "L"),
    ("Teoscar Hernandez",     606466, "16624",     "LAD", ["OF"],          "R", "R"),
    ("Jarren Duran",          680776, "22621",     "BOS", ["OF"],          "L", "L"),
    ("Marcell Ozuna",         542303, "11477",     "ATL", ["OF"],          "R", "R"),
    ("Wyatt Langford",        691234, "25764b",    "TEX", ["OF"],          "R", "R"),
    ("Randy Arozarena",       668227, "22027",     "SEA", ["OF"],          "R", "R"),
    ("Anthony Santander",     606061, "16164",     "TOR", ["OF"],          "S", "R"),
    # ── SP ────────────────────────────────────────────────────────────────────
    ("Zack Wheeler",          554430, "12716",     "PHI", ["SP"],          "R", "R"),
    ("Corbin Burnes",         669203, "21538",     "BAL", ["SP"],          "R", "R"),
    ("Gerrit Cole",           543037, "10155",     "NYY", ["SP"],          "R", "R"),
    ("Dylan Cease",           656302, "21163",     "SD",  ["SP"],          "R", "R"),
    ("Aaron Nola",            605400, "15594",     "PHI", ["SP"],          "R", "R"),
    ("Kevin Gausman",         592332, "14798",     "TOR", ["SP"],          "R", "R"),
    ("Logan Webb",            657277, "21461",     "SF",  ["SP"],          "R", "R"),
    ("Pablo Lopez",           641154, "20663",     "MIN", ["SP"],          "R", "R"),
    ("Yoshinobu Yamamoto",    808963, None,        "LAD", ["SP"],          "R", "R"),
    ("Max Fried",             594789, "13109",     "NYY", ["SP"],          "L", "L"),
    # ── RP ────────────────────────────────────────────────────────────────────
    ("Josh Hader",            624578, "17626",     "HOU", ["RP"],          "L", "L"),
    ("Emmanuel Clase",        667555, "21463",     "CLE", ["RP"],          "R", "R"),
    ("Ryan Helsley",          651529, "21090",     "STL", ["RP"],          "R", "R"),
    ("David Bednar",          676969, "22268",     "PIT", ["RP"],          "R", "R"),
    ("Alexis Diaz",           672515, None,        "CIN", ["RP"],          "R", "R"),
    ("Jordan Romano",         669270, None,        "TOR", ["RP"],          "R", "R"),
    ("Pete Fairbanks",        656432, None,        "TB",  ["RP"],          "R", "R"),
    ("Camilo Doval",          672717, None,        "SF",  ["RP"],          "R", "R"),
]


# ── Roster-slot assignment ────────────────────────────────────────────────────

_ACTIVE_SLOTS = [
    "C", "1B", "2B", "3B", "SS",
    "OF", "OF", "OF",
    "Util", "Util",   # any non-pitcher
    "SP", "SP",
    "RP", "RP",
    "P", "P",         # SP or RP
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
    """Greedy slot assignment. Returns [(slot, player), ...]."""
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
    # Inclusive buckets — multi-position players appear in all relevant buckets.
    # pick_unique() deduplicates via the 'used' set.
    catchers   = [p for p in _PLAYER_POOL if "C"  in p[4]]
    first_base = [p for p in _PLAYER_POOL if "1B" in p[4]]
    second_base= [p for p in _PLAYER_POOL if "2B" in p[4]]
    third_base = [p for p in _PLAYER_POOL if "3B" in p[4]]
    shortstop  = [p for p in _PLAYER_POOL if "SS" in p[4]]
    outfield   = [p for p in _PLAYER_POOL if "OF" in p[4]]
    starters   = [p for p in _PLAYER_POOL if "SP" in p[4]]
    relievers  = [p for p in _PLAYER_POOL if "RP" in p[4]]

    used: set[int] = set()  # mlb_ids already assigned

    def pick_unique(pool: list, n: int) -> list:
        available = [p for p in pool if p[1] not in used]
        rng.shuffle(available)
        chosen = available[:n]
        for p in chosen:
            used.add(p[1])
        return chosen

    def build_pool() -> list[tuple]:
        team: list[tuple] = []
        team += pick_unique(catchers,    1)
        team += pick_unique(first_base,  1)
        team += pick_unique(second_base, 1)
        team += pick_unique(third_base,  1)
        team += pick_unique(shortstop,   1)
        team += pick_unique(outfield,    5)   # 3 OF starters + 2 Util/BN
        team += pick_unique(starters,    3)   # 2 SP starters + 1 P or BN
        team += pick_unique(relievers,   3)   # 2 RP starters + 1 P or BN
        # Fill remainder of BN slots from whatever's left
        leftovers = (
            [p for p in outfield   if p[1] not in used] +
            [p for p in first_base if p[1] not in used] +
            [p for p in second_base if p[1] not in used]
        )
        team += pick_unique(leftovers, 4)
        return team

    all_slots = _ACTIVE_SLOTS + _BENCH_SLOTS
    my_pool  = build_pool()
    opp_pool = build_pool()

    my_roster  = _assign_roster(my_pool,  all_slots)
    opp_roster = _assign_roster(opp_pool, all_slots)
    return my_roster, opp_roster


# ── Stat generation ───────────────────────────────────────────────────────────

_ELITE_BATTERS  = {592450, 660670, 663656, 665742, 670541, 677951, 608369, 608070, 518692}
_ELITE_STARTERS = {554430, 669203, 543037, 808963}

_BATTER_ELITE = dict(ab=4.0, h=1.35, hr=0.20, sb=0.14, bb=0.50, hbp=0.03, sf=0.02,
                     tb=2.1, errors=0.01, chances=3.0)
_BATTER_AVG   = dict(ab=3.2, h=0.85, hr=0.08, sb=0.07, bb=0.32, hbp=0.02, sf=0.015,
                     tb=1.2, errors=0.02, chances=2.5)
# Pitchers: values represent a 3-day week window (not per-start)
_SP_ELITE = dict(ip=12.0, w=0.7, k=18.0, walks_allowed=3.0, hits_allowed=9.0,  sv=0, holds=0)
_SP_AVG   = dict(ip=10.0, w=0.5, k=14.0, walks_allowed=4.0, hits_allowed=11.0, sv=0, holds=0)
_RP_AVG   = dict(ip=2.0,  w=0.1, k=3.0,  walks_allowed=0.7, hits_allowed=1.5,  sv=1.0, holds=0.7)


def _jitter(v: float, rng: random.Random, lo: float = 0.6, hi: float = 1.4) -> float:
    return max(0.0, v * rng.uniform(lo, hi))


def _gen_batter_stats(mlb_id: int, rng: random.Random) -> dict:
    tmpl = _BATTER_ELITE if mlb_id in _ELITE_BATTERS else _BATTER_AVG
    ab = round(_jitter(tmpl["ab"], rng))
    h  = min(round(_jitter(tmpl["h"], rng)), ab)
    hr = min(round(_jitter(tmpl["hr"], rng)), h)
    sb = round(_jitter(tmpl["sb"], rng))
    bb = round(_jitter(tmpl["bb"], rng))
    hbp = round(_jitter(tmpl["hbp"], rng))
    sf  = round(_jitter(tmpl["sf"], rng))
    tb  = max(h, round(_jitter(tmpl["tb"], rng)))
    chances = round(_jitter(tmpl["chances"], rng))
    errors  = min(round(_jitter(tmpl["errors"], rng)), chances)
    avg  = round(h / ab,           3) if ab > 0     else 0.0
    obp  = round((h+bb+hbp) / (ab+bb+hbp+sf), 3) if (ab+bb+hbp+sf) > 0 else 0.0
    slg  = round(tb / ab,          3) if ab > 0     else 0.0
    ops  = round(obp + slg,        3)
    fpct = round((chances-errors) / chances, 3)     if chances > 0  else 1.0
    return dict(
        ab=ab, h=h, hr=hr, sb=sb, bb=bb, hbp=hbp, sf=sf, tb=tb,
        errors=errors, chances=chances, avg=avg, ops=ops, fpct=fpct,
        ip=None, w=None, k=None, walks_allowed=None, hits_allowed=None,
        sv=None, holds=None, whip=None, k_bb=None, sv_h=None,
    )


def _gen_pitcher_stats(mlb_id: int, positions: list[str], rng: random.Random) -> dict:
    if "SP" in positions:
        tmpl = _SP_ELITE if mlb_id in _ELITE_STARTERS else _SP_AVG
    else:
        tmpl = _RP_AVG
    ip = round(_jitter(tmpl["ip"], rng, 0.7, 1.3), 1)
    w  = round(_jitter(tmpl["w"],  rng))
    k  = round(_jitter(tmpl["k"],  rng))
    bb = round(_jitter(tmpl["walks_allowed"], rng))
    ha = round(_jitter(tmpl["hits_allowed"],  rng))
    sv = round(_jitter(tmpl["sv"],   rng))
    hd = round(_jitter(tmpl["holds"], rng))
    sv_h = sv + hd
    whip = round((bb + ha) / ip,  3) if ip > 0  else 0.0
    k_bb = round(k / bb,          2) if bb > 0  else float(k)
    return dict(
        ip=ip, w=w, k=k, walks_allowed=bb, hits_allowed=ha, sv=sv, holds=hd,
        sv_h=sv_h, whip=whip, k_bb=k_bb,
        ab=None, h=None, hr=None, sb=None, bb=None, hbp=None, sf=None, tb=None,
        errors=None, chances=None, avg=None, ops=None, fpct=None,
    )


# ── DB population ─────────────────────────────────────────────────────────────


def populate_db(
    conn: duckdb.DuckDBPyConnection,
    my_roster: list,
    opp_roster: list,
    rng: random.Random,
) -> None:
    create_all_tables(conn)

    # Collect unique players across both rosters
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

    # ── fact_rosters ───────────────────────────────────────────────────────────
    roster_rows = [
        {
            "team_id": team_key,
            "player_id": f"422.p.{player[1]}",
            "snapshot_date": RUN_DATE,
            "roster_slot": slot,
            "acquisition_type": "draft",
        }
        for team_key, assignments in [(MY_TEAM_KEY, my_roster), (OPP_TEAM_KEY, opp_roster)]
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

    # ── fact_player_stats_daily (3 game-days per player) ──────────────────────
    stat_rows = []
    for name, mlb_id, *_, positions, _, _ in players:
        player_id = f"422.p.{mlb_id}"
        is_pitcher = all(pos in ("SP", "RP") for pos in positions)
        for stat_date in STAT_DATES:
            row = (
                _gen_pitcher_stats(mlb_id, positions, rng)
                if is_pitcher
                else _gen_batter_stats(mlb_id, rng)
            )
            row["player_id"] = player_id
            row["stat_date"] = stat_date
            stat_rows.append(row)
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

    # ── fact_matchups (week 21, 2025: my team vs opponent) ────────────────────
    LEAGUE_ID = 87941
    WEEK = 21
    SEASON = 2025
    matchup_id = f"{LEAGUE_ID}_{SEASON}_W{WEEK:02d}_T3vsT5"
    conn.execute(f"""
        INSERT OR REPLACE INTO {FACT_MATCHUPS}
            (matchup_id, league_id, week_number, season, team_id_home, team_id_away)
        VALUES (?, ?, ?, ?, ?, ?)
    """, [matchup_id, LEAGUE_ID, WEEK, SEASON, MY_TEAM_KEY, OPP_TEAM_KEY])

    print(
        f"\n  DB populated: {len(players)} players, {len(stat_rows)} stat rows "
        f"({len(my_roster)} my-team slots, {len(opp_roster)} opp-team slots)"
    )


# ── Pretty printers ────────────────────────────────────────────────────────────


def print_roster(label: str, assignments: list) -> None:
    print(f"\n{'─'*62}")
    print(f"  {label}")
    print(f"{'─'*62}")
    print(f"  {'Slot':<8}  {'Player':<24}  {'Pos':<12}  {'Team'}")
    print(f"  {'─'*56}")
    for slot, (name, _, _, team, positions, _, _) in assignments:
        pos_str = "/".join(positions)
        print(f"  {slot:<8}  {name:<24}  {pos_str:<12}  {team}")


def print_report(report: dict, conn: duckdb.DuckDBPyConnection) -> None:
    print(f"\n{'═'*62}")
    print(f"  DAILY REPORT — {report.get('report_date', '?')}  (Week {report.get('week_number', '?')})")
    print(f"{'═'*62}")

    # ── Matchup summary ────────────────────────────────────────────────────────
    matchup = report.get("matchup_summary", [])
    if matchup:
        wins   = [r["category"].upper() for r in matchup if r.get("my_leads")]
        losses = [r["category"].upper() for r in matchup if not r.get("my_leads") and r.get("status") != "toss_up"]
        ties   = [r["category"].upper() for r in matchup if r.get("status") == "toss_up"]

        print(f"\n  Matchup vs Opponent")
        print(f"  {'Category':<8}  {'Mine':>8}  {'Opp':>8}  {'Status'}")
        print(f"  {'─'*46}")
        for row in matchup:
            cat    = row["category"].upper()
            mine   = row.get("my_value", 0)
            opp    = row.get("opp_value", 0)
            status = row.get("status", "?")
            symbol = "✓" if row.get("my_leads") else ("✗" if status in ("safe_loss", "losing") else "~")
            print(f"  {symbol} {cat:<7}  {mine:>8.3f}  {opp:>8.3f}  {status}")
        print(f"\n  Projected: leading {len(wins)} cats, trailing {len(losses)}, tied {len(ties)}")

    # ── Lineup ─────────────────────────────────────────────────────────────────
    lineup = report.get("lineup", {})
    if lineup:
        # Resolve player_ids → names from dim_players
        pid_list = list(lineup.values())
        if pid_list:
            placeholders = ", ".join("?" * len(pid_list))
            name_map = dict(
                conn.execute(
                    f"SELECT player_id, full_name FROM {DIM_PLAYERS} WHERE player_id IN ({placeholders})",
                    pid_list,
                ).fetchall()
            )
        else:
            name_map = {}
        print(f"\n  Recommended Lineup")
        print(f"  {'─'*40}")
        for slot, pid in lineup.items():
            name = name_map.get(pid, pid)
            print(f"  {slot:<8}  {name}")
    else:
        print("\n  Lineup: (no lineup generated — no position eligibility data)")

    # ── Waiver adds ────────────────────────────────────────────────────────────
    adds = report.get("adds", [])
    if adds:
        print(f"\n  Waiver Wire Recommendations ({len(adds)}):")
        for a in adds[:5]:
            print(f"  +  {a.get('add_player_id', '?')}  (drop {a.get('drop_player_id', '?')})")
            print(f"       reason: {a.get('reason', '?')}")

    # ── IP pace ────────────────────────────────────────────────────────────────
    ip = report.get("ip_pace", {})
    if ip:
        cur  = float(ip.get("current_ip", 0))
        proj = float(ip.get("projected_ip", 0))
        need = int(ip.get("min_ip", 21))
        on   = ip.get("on_pace", False)
        sym  = "✓" if on else "⚠ "
        print(f"\n  IP Pace: {cur:.1f} IP so far, {proj:.1f} projected, need {need} — {sym}")

    # ── Callup alerts ──────────────────────────────────────────────────────────
    alerts = report.get("callup_alerts", [])
    if alerts:
        print(f"\n  Callup Alerts ({len(alerts)}):")
        for a in alerts:
            print(f"  ↑ {a.get('player_name', '?')} ({a.get('team', '?')}) — {a.get('days_since_callup', '?')} days ago")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="2025 test pipeline run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    parser.add_argument("--json", action="store_true", help="Also dump raw report JSON")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    rng = random.Random(args.seed)

    print(f"\n{'═'*62}")
    print(f"  2025 Fantasy Baseball — End-to-End Pipeline Test")
    print(f"  Run date  : {RUN_DATE}  (fantasy week 21, regular season)")
    print(f"  Stat dates: {STAT_DATES[0]} → {STAT_DATES[-1]}")
    print(f"  Seed      : {args.seed}")
    print(f"{'═'*62}")

    # Assign rosters
    my_roster, opp_roster = build_rosters(rng)
    print_roster(f"MY TEAM  ({MY_TEAM_KEY})", my_roster)
    print_roster(f"OPPONENT ({OPP_TEAM_KEY})", opp_roster)

    # Populate in-memory DB
    conn = duckdb.connect(":memory:")
    populate_db(conn, my_roster, opp_roster, rng)

    # Load settings and inject team key
    settings = dataclasses.replace(
        load_league_settings(), my_team_key=MY_TEAM_KEY
    )

    # Give all players a game today (schedule_df)
    all_player_ids = conn.execute(
        f"SELECT player_id FROM {DIM_PLAYERS}"
    ).fetchdf()[["player_id"]]

    # Run pipeline — mock only the external API steps
    with (
        mock.patch(
            "src.pipeline.daily_run._step_load_yahoo",
            return_value=({}, pd.DataFrame()),
        ),
        mock.patch(
            "src.pipeline.daily_run._step_load_mlb_stats",
            return_value=({}, all_player_ids),
        ),
        mock.patch(
            "src.pipeline.daily_run._step_refresh_projections",
            return_value={},
        ),
    ):
        print("\n  Running pipeline...")
        result = run_daily_pipeline(conn, settings, run_date=RUN_DATE)

    status = result["status"]
    rows   = result["rows_written"]
    print(f"  Status      : {status}")
    print(f"  Rows written: {rows}")

    # Read back report
    row = conn.execute(
        "SELECT report_json FROM fact_daily_reports WHERE report_date = ?",
        [RUN_DATE],
    ).fetchone()

    if row and row[0]:
        report = json.loads(row[0]) if isinstance(row[0], str) else row[0]
        print_report(report, conn)
        if args.json:
            print("\n── Raw JSON ──────────────────────────────────────────────")
            print(json.dumps(report, indent=2, default=str))
    else:
        print("\n  [No report in DB]")

    conn.close()


if __name__ == "__main__":
    main()
