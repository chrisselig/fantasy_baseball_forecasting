"""
server.py

Reactive server logic for the Shiny app.
All expensive operations go in @reactive.calc (cached).
Stub data is used when DB is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import duckdb
import pandas as pd
from htmltools import Tag
from shiny import Inputs, Outputs, Session, reactive, render, ui

from src.app.stubs import STUB_DAILY_REPORT, STUB_ROSTER_DF, STUB_WAIVER_DF
from src.db.connection import managed_connection
from src.db.schema import (
    DIM_PLAYERS,
    FACT_DAILY_REPORTS,
    FACT_ROSTERS,
    FACT_WAIVER_SCORES,
)

logger = logging.getLogger(__name__)

# League ID for constructing team key when my_team_key is not in config
_LEAGUE_ID = 87941


def _get_my_team_key() -> str:
    """Return my Yahoo team key from env var.

    Reads YAHOO_TEAM_KEY directly if set, otherwise constructs from
    YAHOO_TEAM_ID (integer part only).
    """
    full_key = os.environ.get("YAHOO_TEAM_KEY", "")
    if full_key:
        return full_key
    team_id = os.environ.get("YAHOO_TEAM_ID", "")
    if team_id:
        return f"422.l.{_LEAGUE_ID}.t.{team_id}"
    return ""


def _load_data_freshness() -> dict[str, object]:
    """Query when the last successful pipeline run completed.

    Returns:
        {
          "generated_at": str | None,   # ISO timestamp or None
          "is_stale": bool,             # True if > 24 hours old
          "is_offline": bool,           # True if DB query failed
        }
    """
    try:
        with managed_connection() as conn:
            row = conn.execute("""
                SELECT generated_at
                FROM fact_daily_reports
                WHERE report_date = CURRENT_DATE
                LIMIT 1
            """).fetchone()
            if row and row[0]:
                import datetime

                generated_at = str(row[0])
                # Parse and check staleness
                try:
                    ts = datetime.datetime.fromisoformat(generated_at)
                    age = datetime.datetime.now() - ts
                    is_stale = age.total_seconds() > 86400  # > 24 hours
                except ValueError:
                    is_stale = False
                return {
                    "generated_at": generated_at,
                    "is_stale": is_stale,
                    "is_offline": False,
                }
    except Exception as exc:
        logger.warning("Could not load data freshness: %s", exc)
    return {"generated_at": None, "is_stale": False, "is_offline": True}


def _load_daily_report() -> dict[str, Any]:
    """Load today's pre-built report from MotherDuck.

    Queries fact_daily_reports for today's report_json.
    Falls back to stub data if DB is unavailable or no report exists yet.
    """
    try:
        with managed_connection() as conn:
            result = conn.execute(f"""
                SELECT report_json
                FROM {FACT_DAILY_REPORTS}
                WHERE report_date = CURRENT_DATE
                LIMIT 1
            """).fetchone()
            if result and result[0]:
                return dict(json.loads(str(result[0])))
    except (duckdb.Error, Exception) as exc:
        logger.warning("Could not load daily report from DB: %s", exc)
    return dict(STUB_DAILY_REPORT)


def _load_roster() -> pd.DataFrame:
    """Load current roster from MotherDuck.

    Queries fact_rosters joined with dim_players for today's snapshot.
    Falls back to stub data if DB is unavailable.
    """
    team_key = _get_my_team_key()
    if not team_key:
        logger.warning("YAHOO_TEAM_KEY / YAHOO_TEAM_ID not set — using stub roster.")
        return STUB_ROSTER_DF.copy()

    try:
        with managed_connection() as conn:
            df: pd.DataFrame = conn.execute(
                f"""
                SELECT
                    r.roster_slot        AS slot,
                    p.full_name          AS player_name,
                    p.team,
                    array_to_string(p.positions, ',') AS position,
                    COALESCE(s.h,  0)    AS h,
                    COALESCE(s.hr, 0)    AS hr,
                    COALESCE(s.sb, 0)    AS sb,
                    COALESCE(s.bb, 0)    AS bb,
                    COALESCE(s.avg, 0.0) AS avg,
                    COALESCE(s.ops, 0.0) AS ops,
                    COALESCE(s.w,  0)    AS w,
                    COALESCE(s.k,  0)    AS k,
                    COALESCE(s.whip, 0.0) AS whip,
                    COALESCE(s.k_bb, 0.0) AS k_bb,
                    COALESCE(s.sv_h, 0)  AS sv_h
                FROM {FACT_ROSTERS} r
                LEFT JOIN {DIM_PLAYERS} p
                    ON r.player_id = p.player_id
                LEFT JOIN (
                    SELECT
                        player_id,
                        SUM(h) AS h, SUM(hr) AS hr, SUM(sb) AS sb, SUM(bb) AS bb,
                        SUM(ab) AS ab, SUM(hbp) AS hbp, SUM(sf) AS sf, SUM(tb) AS tb,
                        SUM(ip) AS ip, SUM(w) AS w, SUM(k) AS k,
                        SUM(walks_allowed) AS walks_allowed,
                        SUM(hits_allowed) AS hits_allowed,
                        SUM(sv) AS sv, SUM(holds) AS holds,
                        CASE WHEN SUM(ab) > 0
                             THEN CAST(SUM(h) AS DOUBLE) / SUM(ab) ELSE 0 END AS avg,
                        CASE WHEN SUM(ab) > 0
                             THEN (CAST(SUM(h)+SUM(bb)+SUM(hbp) AS DOUBLE)
                                   / NULLIF(SUM(ab)+SUM(bb)+SUM(hbp)+SUM(sf), 0))
                                  + CAST(SUM(tb) AS DOUBLE) / SUM(ab)
                             ELSE 0 END AS ops,
                        CASE WHEN SUM(ip) > 0
                             THEN CAST(SUM(walks_allowed)+SUM(hits_allowed) AS DOUBLE)
                                  / SUM(ip) ELSE 0 END AS whip,
                        CASE WHEN SUM(walks_allowed) > 0
                             THEN CAST(SUM(k) AS DOUBLE) / SUM(walks_allowed)
                             ELSE 0 END AS k_bb,
                        SUM(sv) + SUM(holds) AS sv_h
                    FROM fact_player_stats_daily
                    WHERE stat_date >= DATE_TRUNC('week', CURRENT_DATE)
                    GROUP BY player_id
                ) s ON r.player_id = s.player_id
                WHERE r.team_id = ?
                  AND r.snapshot_date = CURRENT_DATE
                ORDER BY r.roster_slot
            """,
                [team_key],
            ).fetchdf()

            if not df.empty:
                return df
    except (duckdb.Error, Exception) as exc:
        logger.warning("Could not load roster from DB: %s", exc)
    return STUB_ROSTER_DF.copy()


def _load_waiver_data() -> pd.DataFrame:
    """Load waiver wire rankings from MotherDuck.

    Queries fact_waiver_scores joined with dim_players for today's scores.
    Falls back to yesterday's scores if today's aren't available yet.
    Falls back to stub data if DB is unavailable.
    """
    try:
        with managed_connection() as conn:
            df: pd.DataFrame = conn.execute(f"""
                SELECT
                    ROW_NUMBER() OVER (ORDER BY w.overall_score DESC) AS rank,
                    p.full_name          AS player_name,
                    p.team,
                    array_to_string(p.positions, ',') AS position,
                    w.overall_score      AS score,
                    w.is_callup          AS callup,
                    NULL                 AS from_level,
                    w.days_since_callup
                FROM {FACT_WAIVER_SCORES} w
                JOIN {DIM_PLAYERS} p ON w.player_id = p.player_id
                WHERE w.score_date = (
                    SELECT MAX(score_date)
                    FROM {FACT_WAIVER_SCORES}
                    WHERE overall_score > 0
                )
                  AND w.overall_score > 0
                ORDER BY w.overall_score DESC
            """).fetchdf()

            if not df.empty:
                return df
    except (duckdb.Error, Exception) as exc:
        logger.warning("Could not load waiver data from DB: %s", exc)
    return STUB_WAIVER_DF.copy()


def _fmt_stat(v: Any) -> str:
    """Format a stat value for display."""
    if isinstance(v, float):
        return f"{v:.3f}" if v < 10 else str(int(v))
    return str(v)


def server(input: Inputs, output: Outputs, session: Session) -> None:
    """Shiny server function wiring reactive calcs and output renderers."""

    # ── Refresh trigger ───────────────────────────────────────────────────────
    _refresh_counter: reactive.Value[int] = reactive.Value(0)

    # ── Reactive data sources ──────────────────────────────────────────────────

    @reactive.calc
    def data_freshness() -> dict[str, object]:
        """Data freshness metadata."""
        _refresh_counter()  # depend on refresh trigger
        return _load_data_freshness()

    @render.ui
    def data_status_banner() -> Tag:
        """Show data freshness / offline status banner at top of every tab."""
        freshness = data_freshness()
        is_offline = bool(freshness.get("is_offline", False))
        is_stale = bool(freshness.get("is_stale", False))
        generated_at = freshness.get("generated_at")

        if is_offline:
            return ui.div(
                ui.span(
                    "\u26a0\ufe0f Offline \u2014 showing cached data. Connect to MotherDuck to refresh."
                ),
                class_="alert alert-danger mb-0 py-1",
                role="alert",
            )
        elif is_stale:
            return ui.div(
                ui.span(
                    f"\u26a0\ufe0f Data may be stale. Last updated: {generated_at}"
                ),
                class_="alert alert-warning mb-0 py-1",
                role="alert",
            )
        elif generated_at:
            return ui.div(
                ui.span(f"\u2705 Data as of {generated_at}"),
                class_="alert alert-success mb-0 py-1",
                role="alert",
            )
        return ui.div()

    @reactive.calc
    def daily_report() -> dict[str, Any]:
        """Cached daily report dict."""
        _refresh_counter()  # depend on refresh trigger
        return _load_daily_report()

    @reactive.calc
    def matchup_data() -> pd.DataFrame:
        """Matchup summary as a tidy DataFrame."""
        report = daily_report()
        rows = report.get("matchup_summary", [])
        if not isinstance(rows, list):
            return pd.DataFrame()
        return pd.DataFrame(rows)

    @reactive.calc
    def roster_data() -> pd.DataFrame:
        """Current roster DataFrame."""
        _refresh_counter()  # depend on refresh trigger
        return _load_roster()

    @reactive.calc
    def waiver_data() -> pd.DataFrame:
        """Filtered waiver wire DataFrame."""
        _refresh_counter()  # depend on refresh trigger
        df = _load_waiver_data()

        pos = str(input.position_filter())
        if pos and pos != "All":
            df = df[df["position"].str.contains(pos, na=False)]

        callup_only: bool = bool(input.callup_only())
        if callup_only:
            df = df[df["callup"].astype(bool)]

        return df.reset_index(drop=True)

    # ── Outputs ────────────────────────────────────────────────────────────────

    # Tab 1: Dashboard

    @render.text
    def header_date() -> str:
        report = daily_report()
        return str(report.get("report_date", "\u2014"))

    @render.text
    def header_week() -> str:
        report = daily_report()
        return f"Week {report.get('week_number', '\u2014')}"

    @render.text
    def header_ip_pace() -> str:
        report = daily_report()
        ip_pace = report.get("ip_pace", {})
        if not isinstance(ip_pace, dict):
            return "\u2014"
        current = ip_pace.get("current_ip", 0.0)
        projected = ip_pace.get("projected_ip", 0.0)
        min_ip = ip_pace.get("min_ip", 21)
        on_pace = ip_pace.get("on_pace", False)
        indicator = "\u2713" if on_pace else "\u26a0"
        return f"{indicator} {current}/{min_ip} (proj {projected})"

    @render.data_frame
    def lineup_table() -> pd.DataFrame:
        report = daily_report()
        lineup = report.get("lineup", {})
        if not isinstance(lineup, dict):
            return pd.DataFrame(columns=["Slot", "Player ID"])
        rows = [{"Slot": slot, "Player ID": pid} for slot, pid in lineup.items()]
        return pd.DataFrame(rows)

    @render.data_frame
    def matchup_scoreboard() -> pd.DataFrame:
        df = matchup_data()
        if df.empty:
            return pd.DataFrame(columns=["Category", "Mine", "Opponent", "Win%"])
        result = pd.DataFrame(
            {
                "Category": df["category"].str.upper(),
                "Mine": df["my_value"].apply(_fmt_stat),
                "Opponent": df["opp_value"].apply(_fmt_stat),
                "Win%": df["win_prob"].apply(lambda p: f"{p * 100:.0f}%"),
                "Status": df["status"],
            }
        )
        return result

    @render.data_frame
    def adds_table() -> pd.DataFrame:
        report = daily_report()
        raw_adds = report.get("adds", [])
        if not isinstance(raw_adds, list) or len(raw_adds) == 0:
            return pd.DataFrame(columns=["Add", "Drop", "Score", "Reason", "Improves"])
        rows: list[dict[str, Any]] = []
        for item in raw_adds:
            a: dict[str, Any] = item if isinstance(item, dict) else {}
            cats = a.get("categories_improved", [])
            rows.append(
                {
                    "Add": str(a.get("add_player_id", "")),
                    "Drop": str(a.get("drop_player_id", "")),
                    "Score": a.get("score", ""),
                    "Reason": str(a.get("reason", "")),
                    "Improves": ", ".join(cats)
                    if isinstance(cats, list)
                    else str(cats),
                }
            )
        return pd.DataFrame(rows)

    @render.ui
    def callup_alerts_ui() -> Tag:
        report = daily_report()
        raw_alerts = report.get("callup_alerts", [])
        if not isinstance(raw_alerts, list) or len(raw_alerts) == 0:
            return ui.p("No call-up alerts today.")
        items: list[Any] = []
        for item in raw_alerts:
            alert: dict[str, Any] = item if isinstance(item, dict) else {}
            name = str(alert.get("player_name", "Unknown"))
            team = str(alert.get("team", ""))
            level = str(alert.get("from_level", ""))
            days = alert.get("days_since_callup", 0)
            items.append(
                ui.div(
                    ui.strong(f"{name} ({team})"),
                    ui.span(
                        f" \u2014 Called up from {level}, {days} day(s) ago",
                        style="margin-left: 0.5rem;",
                    ),
                    class_="alert alert-warning mb-2",
                    role="alert",
                )
            )
        return ui.div(*items)

    # Tab 2: Matchup Detail

    @render.data_frame
    def matchup_detail_table() -> pd.DataFrame:
        df = matchup_data()
        if df.empty:
            return pd.DataFrame()
        result = pd.DataFrame(
            {
                "Category": df["category"].str.upper(),
                "My Value": df["my_value"],
                "Opp Value": df["opp_value"],
                "Leading": df["my_leads"].map({True: "Yes", False: "No"}),
                "Margin %": df["margin_pct"].apply(lambda p: f"{p * 100:.1f}%"),
                "Win Prob": df["win_prob"].apply(lambda p: f"{p * 100:.0f}%"),
                "Status": df["status"],
            }
        )
        return result

    # Tab 3: Waiver Wire

    @render.data_frame
    def waiver_table() -> pd.DataFrame:
        df = waiver_data()
        display_cols = [
            "rank",
            "player_name",
            "team",
            "position",
            "score",
            "callup",
            "from_level",
            "days_since_callup",
        ]
        available = [c for c in display_cols if c in df.columns]
        return df[available].rename(
            columns={
                "rank": "Rank",
                "player_name": "Player",
                "team": "Team",
                "position": "Position",
                "score": "Score",
                "callup": "Callup?",
                "from_level": "From",
                "days_since_callup": "Days Since Callup",
            }
        )

    # Tab 4: Roster

    @render.data_frame
    def roster_table() -> pd.DataFrame:
        df = roster_data()
        display_cols = [
            "slot",
            "player_name",
            "team",
            "position",
            "h",
            "hr",
            "sb",
            "bb",
            "avg",
            "ops",
            "w",
            "k",
            "whip",
            "k_bb",
            "sv_h",
        ]
        available = [c for c in display_cols if c in df.columns]
        return df[available].rename(
            columns={
                "slot": "Slot",
                "player_name": "Player",
                "team": "Team",
                "position": "Pos",
                "h": "H",
                "hr": "HR",
                "sb": "SB",
                "bb": "BB",
                "avg": "AVG",
                "ops": "OPS",
                "w": "W",
                "k": "K",
                "whip": "WHIP",
                "k_bb": "K/BB",
                "sv_h": "SV+H",
            }
        )

    # ── Manual refresh ─────────────────────────────────────────────────────────

    @reactive.effect
    @reactive.event(input.refresh_btn)
    def _refresh() -> None:
        _refresh_counter.set(_refresh_counter() + 1)
