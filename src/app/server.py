"""
server.py

Reactive server logic for the Shiny app.
All expensive operations go in @reactive.calc (cached).
Returns empty results when data is unavailable.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import duckdb
import pandas as pd
from htmltools import Tag
from shiny import Inputs, Outputs, Session, reactive, render, ui

from src.analysis.hot_cold import annotate_with_streaks, match_win_probability
from src.config import load_league_settings
from src.db.connection import managed_connection
from src.db.schema import (
    DIM_PLAYERS,
    FACT_DAILY_REPORTS,
    FACT_MATCHUPS,
    FACT_PLAYER_NEWS,
    FACT_PLAYER_STATS_DAILY,
    FACT_PROJECTIONS,
    FACT_ROSTERS,
    FACT_TRANSACTIONS,
    FACT_WAIVER_SCORES,
)

logger = logging.getLogger(__name__)

# ── Category metadata ──────────────────────────────────────────────────────

_CATEGORY_META: dict[str, dict[str, str]] = {
    "h": {"label": "H", "desc": "Hits", "type": "batter", "win": "highest"},
    "hr": {"label": "HR", "desc": "Home Runs", "type": "batter", "win": "highest"},
    "sb": {"label": "SB", "desc": "Stolen Bases", "type": "batter", "win": "highest"},
    "bb": {"label": "BB", "desc": "Walks", "type": "batter", "win": "highest"},
    "fpct": {"label": "FPCT", "desc": "Fielding %", "type": "batter", "win": "highest"},
    "avg": {
        "label": "AVG",
        "desc": "Batting Average",
        "type": "batter",
        "win": "highest",
    },
    "ops": {
        "label": "OPS",
        "desc": "On-Base + Slugging",
        "type": "batter",
        "win": "highest",
    },
    "w": {"label": "W", "desc": "Wins", "type": "pitcher", "win": "highest"},
    "k": {"label": "K", "desc": "Strikeouts", "type": "pitcher", "win": "highest"},
    "whip": {"label": "WHIP", "desc": "(W+H)/IP", "type": "pitcher", "win": "lowest"},
    "k_bb": {
        "label": "K/BB",
        "desc": "K / Walk Ratio",
        "type": "pitcher",
        "win": "highest",
    },
    "sv_h": {
        "label": "SV+H",
        "desc": "Saves + Holds",
        "type": "pitcher",
        "win": "highest",
    },
}

_STATUS_CLASS: dict[str, str] = {
    "safe_win": "status-safe-win",
    "flippable_win": "status-flippable",
    "toss_up": "status-toss-up",
    "flippable_loss": "status-toss-up",
    "safe_loss": "status-safe-loss",
}

_STATUS_LABEL: dict[str, str] = {
    "safe_win": "Safe Win",
    "flippable_win": "Leading",
    "toss_up": "Toss-Up",
    "flippable_loss": "Trailing",
    "safe_loss": "Safe Loss",
}

_PITCHER_SLOTS = frozenset({"SP", "SP1", "SP2", "RP", "RP1", "RP2", "P", "P1", "P2"})

_TRANSACTION_TYPE_LABELS: dict[str, tuple[str, str]] = {
    "mlb_injury": ("🩹 Injury", "#c62828"),
    "mlb_activation": ("✅ Activation", "#2e7d32"),
    "mlb_callup": ("⬆ Call-up", "#1a7fa1"),
    "mlb_demotion": ("⬇ Demotion", "#7b5800"),
}


def _win_pct_class(pct: float) -> str:
    """CSS class for win probability colouring."""
    if pct >= 0.65:
        return "win-high"
    if pct >= 0.35:
        return "win-mid"
    return "win-low"


def _html_table(headers: list[str], rows: list[list[Any]]) -> Tag:
    """Build a styled HTML table matching the Savant theme."""
    th_cells = [ui.tags.th(h) for h in headers]
    tbody_rows: list[Any] = []
    for row in rows:
        td_cells = [
            ui.tags.td(cell) if isinstance(cell, Tag) else ui.tags.td(str(cell))
            for cell in row
        ]
        tbody_rows.append(ui.tags.tr(*td_cells))
    return ui.tags.table(
        ui.tags.thead(ui.tags.tr(*th_cells)),
        ui.tags.tbody(*tbody_rows),
        class_="shiny-data-frame table table-sm",
        style="width:100%;",
    )


def _streak_badge(label: str) -> Tag:
    """Styled badge for hot/cold/warm label."""
    if label == "🔥 Hot":
        style = (
            "background:#d32f2f;color:#fff;padding:1px 7px;"
            "border-radius:10px;font-size:0.72rem;"
        )
    elif label == "☀️ Warm":
        style = (
            "background:#e65100;color:#fff;padding:1px 7px;"
            "border-radius:10px;font-size:0.72rem;"
        )
    elif label == "❄️ Cold":
        style = (
            "background:#1565c0;color:#fff;padding:1px 7px;"
            "border-radius:10px;font-size:0.72rem;"
        )
    else:
        style = "color:#888;font-size:0.72rem;"
    return ui.tags.span(label, style=style)


def _stat_box(title: str, value: str, sub: str = "", color: str = "#1a7fa1") -> Tag:
    """Compact dark stat box used in summary rows."""
    return ui.tags.div(
        ui.tags.span(
            title,
            style="font-size:0.65rem;font-weight:700;text-transform:uppercase;"
            "letter-spacing:0.08em;color:#7ab8d4;display:block;",
        ),
        ui.tags.div(
            ui.tags.span(value, style="font-size:1.1rem;font-weight:700;color:#fff;"),
            ui.tags.span(f"  {sub}", style="font-size:0.78rem;color:#8ab;")
            if sub
            else ui.tags.span(),
        ),
        style=f"background:#132747;border-radius:4px;border-left:3px solid {color};"
        "padding:0.5rem 0.75rem;height:100%;",
    )


# Categories that should display as integers (no decimal places).
_INTEGER_CATS: set[str] = {"h", "hr", "sb", "bb", "w", "k", "sv_h"}

# Categories that should display with exactly 2 decimal places.
_TWO_DECIMAL_CATS: set[str] = {"whip", "k_bb"}

# Categories that should display with exactly 3 decimal places.
_THREE_DECIMAL_CATS: set[str] = {"avg", "ops", "fpct"}


def _fmt_stat(v: Any, cat: str = "") -> str:
    """Format a stat value for display. Returns '—' for None/NaN.

    Args:
        v: The stat value to format.
        cat: Lowercase category key (e.g. "hr", "whip", "avg").
             When provided, formatting is category-aware.
    """
    if v is None:
        return "—"
    if isinstance(v, float):
        import math

        if math.isnan(v):
            return "—"
        cat_lower = cat.lower()
        if cat_lower in _INTEGER_CATS:
            return str(int(round(v)))
        if cat_lower in _TWO_DECIMAL_CATS:
            return f"{v:.2f}"
        if cat_lower in _THREE_DECIMAL_CATS:
            return f"{v:.3f}"
        # Fallback: 3 decimals for small values, integer for large
        return f"{v:.3f}" if v < 10 else str(int(v))
    return str(v)


def _safe_count(v: Any) -> str:
    """Format an integer counting stat, returning '—' for None/NaN."""
    if v is None:
        return "—"
    if isinstance(v, float):
        import math

        if math.isnan(v):
            return "—"
        return str(int(v))
    return str(v)


# ── Data loaders ───────────────────────────────────────────────────────────


def _get_my_team_key() -> str:
    """Return my team key from league config."""
    return load_league_settings().my_team_key


# Yahoo scoreboard column → category key mapping
_YAHOO_STAT_MAP_HOME: dict[str, str] = {
    "h_home": "h",
    "hr_home": "hr",
    "sb_home": "sb",
    "bb_home": "bb",
    "fpct_home": "fpct",
    "avg_home": "avg",
    "ops_home": "ops",
    "w_home": "w",
    "k_home": "k",
    "whip_home": "whip",
    "k_bb_home": "k_bb",
    "sv_h_home": "sv_h",
}
_YAHOO_STAT_MAP_AWAY: dict[str, str] = {
    "h_away": "h",
    "hr_away": "hr",
    "sb_away": "sb",
    "bb_away": "bb",
    "fpct_away": "fpct",
    "avg_away": "avg",
    "ops_away": "ops",
    "w_away": "w",
    "k_away": "k",
    "whip_away": "whip",
    "k_bb_away": "k_bb",
    "sv_h_away": "sv_h",
}


def _load_yahoo_matchup_stats(week: int | None = None) -> dict[str, dict[str, float]]:
    """Load actual weekly stats from Yahoo scoreboard (fact_matchups).

    Args:
        week: If provided, load stats for that specific week.
              If None, load the most recent week.

    Returns a dict ``{"mine": {cat: val, ...}, "opp": {cat: val, ...}}``.
    """
    empty: dict[str, dict[str, float]] = {"mine": {}, "opp": {}}
    try:
        team_key = _get_my_team_key()
        with managed_connection() as conn:
            if week is not None:
                row = conn.execute(
                    f"""
                    SELECT *
                    FROM {FACT_MATCHUPS}
                    WHERE (team_id_home = ? OR team_id_away = ?)
                      AND week_number = ?
                    ORDER BY week_number DESC
                    LIMIT 1
                """,
                    [team_key, team_key, week],
                ).fetchone()
            else:
                row = conn.execute(
                    f"""
                    SELECT *
                    FROM {FACT_MATCHUPS}
                    WHERE (team_id_home = ? OR team_id_away = ?)
                    ORDER BY week_number DESC
                    LIMIT 1
                """,
                    [team_key, team_key],
                ).fetchone()
            if not row:
                return empty
            cols = [
                desc[0]
                for desc in conn.execute(
                    f"SELECT * FROM {FACT_MATCHUPS} LIMIT 0"
                ).description
            ]
            data = dict(zip(cols, row, strict=False))

            is_home = data.get("team_id_home") == team_key
            mine_map = _YAHOO_STAT_MAP_HOME if is_home else _YAHOO_STAT_MAP_AWAY
            opp_map = _YAHOO_STAT_MAP_AWAY if is_home else _YAHOO_STAT_MAP_HOME

            mine: dict[str, float] = {}
            opp: dict[str, float] = {}
            for col, cat in mine_map.items():
                v = data.get(col)
                if v is not None:
                    try:
                        mine[cat] = float(v)
                    except (ValueError, TypeError):
                        pass
            for col, cat in opp_map.items():
                v = data.get(col)
                if v is not None:
                    try:
                        opp[cat] = float(v)
                    except (ValueError, TypeError):
                        pass
            return {"mine": mine, "opp": opp}
    except Exception as exc:
        logger.warning("Could not load Yahoo matchup stats: %s", exc)
        return empty


def _load_data_freshness() -> dict[str, object]:
    """Query when the last successful pipeline run completed."""
    try:
        with managed_connection() as conn:
            row = conn.execute("""
                SELECT report_date, generated_at
                FROM fact_daily_reports
                ORDER BY report_date DESC
                LIMIT 1
            """).fetchone()
            if row and row[0]:
                import datetime

                report_date = str(row[0])
                generated_at = str(row[1]) if row[1] else None
                is_stale = report_date != datetime.date.today().isoformat()
                return {
                    "generated_at": generated_at,
                    "report_date": report_date,
                    "is_stale": is_stale,
                    "is_offline": False,
                }
    except Exception as exc:
        logger.warning("Could not load data freshness: %s", exc)
        return {
            "generated_at": None,
            "report_date": None,
            "is_stale": False,
            "is_offline": True,
            "error": str(exc),
        }
    return {
        "generated_at": None,
        "report_date": None,
        "is_stale": False,
        "is_offline": True,
        "error": "No report found in database.",
    }


_EMPTY_ROSTER_COLS: list[str] = [
    "slot",
    "player_id",
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


def _empty_roster_df() -> pd.DataFrame:
    """Return an empty DataFrame with the expected roster columns."""
    return pd.DataFrame(columns=_EMPTY_ROSTER_COLS)


def _load_available_weeks() -> dict[str, str]:
    """Return ``{week_number_str: label, ...}`` for all weeks with reports.

    Only includes weeks 1–26 (valid fantasy weeks) and orders them ascending
    so the dropdown reads naturally.
    """
    weeks: dict[str, str] = {"latest": "Latest"}
    try:
        with managed_connection() as conn:
            import datetime as _dt

            current_season = _dt.date.today().year
            rows = conn.execute(
                f"""
                SELECT DISTINCT week_number
                FROM {FACT_DAILY_REPORTS}
                WHERE week_number BETWEEN 1 AND 26
                  AND season = ?
                ORDER BY week_number ASC
            """,
                [current_season],
            ).fetchall()
            for (wk,) in rows:
                weeks[str(wk)] = f"Week {wk}"
    except (duckdb.Error, Exception) as exc:
        logger.warning("Could not load available weeks: %s", exc)
    return weeks


def _load_daily_report(week: int | None = None) -> dict[str, Any]:
    """Load a pre-built report from MotherDuck.

    Args:
        week: If provided, load the latest report for that week.
              If None, load the most recent report overall.
    """
    try:
        with managed_connection() as conn:
            if week is not None:
                result = conn.execute(
                    f"""
                    SELECT report_json
                    FROM {FACT_DAILY_REPORTS}
                    WHERE week_number = ?
                    ORDER BY report_date DESC
                    LIMIT 1
                """,
                    [week],
                ).fetchone()
            else:
                result = conn.execute(f"""
                    SELECT report_json
                    FROM {FACT_DAILY_REPORTS}
                    ORDER BY report_date DESC
                    LIMIT 1
                """).fetchone()
            if result and result[0]:
                return dict(json.loads(str(result[0])))
    except (duckdb.Error, Exception) as exc:
        logger.warning("Could not load daily report from DB: %s", exc)
    return {}


def _load_recent_daily_stats(window_days: int = 10) -> pd.DataFrame:
    """Load recent daily stats for streak computation."""
    try:
        with managed_connection() as conn:
            df: pd.DataFrame = conn.execute(f"""
                SELECT *
                FROM {FACT_PLAYER_STATS_DAILY}
                WHERE stat_date >= (
                    SELECT MAX(stat_date) - INTERVAL '{window_days} days'
                    FROM {FACT_PLAYER_STATS_DAILY}
                )
                ORDER BY player_id, stat_date
            """).fetchdf()
            return df
    except (duckdb.Error, Exception) as exc:
        logger.warning("Could not load daily stats for streaks: %s", exc)
    return pd.DataFrame()


def _load_roster() -> pd.DataFrame:
    """Load current roster from MotherDuck with streak annotations."""
    team_key = _get_my_team_key()
    if not team_key:
        return _empty_roster_df()

    try:
        with managed_connection() as conn:
            snap_row = conn.execute(
                f"SELECT MAX(snapshot_date) FROM {FACT_ROSTERS} WHERE team_id = ?",
                [team_key],
            ).fetchone()
            if not snap_row or not snap_row[0]:
                return _empty_roster_df()
            latest_snapshot = snap_row[0]

            import datetime as _dt

            week_start = latest_snapshot - _dt.timedelta(days=latest_snapshot.weekday())

            df: pd.DataFrame = conn.execute(
                f"""
                SELECT
                    r.roster_slot        AS slot,
                    p.player_id,
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
                LEFT JOIN {DIM_PLAYERS} p ON r.player_id = p.player_id
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
                    WHERE stat_date >= ?
                    GROUP BY player_id
                ) s ON r.player_id = s.player_id
                WHERE r.team_id = ?
                  AND r.snapshot_date = ?
                ORDER BY r.roster_slot
            """,
                [week_start, team_key, latest_snapshot],
            ).fetchdf()

            if not df.empty:
                daily_df = _load_recent_daily_stats()
                if not daily_df.empty and "player_id" in df.columns:
                    df = annotate_with_streaks(df, daily_df)
                elif "streak" not in df.columns:
                    df["streak"] = "—"
                return df
    except (duckdb.Error, Exception) as exc:
        logger.warning("Could not load roster from DB: %s", exc)
    return _empty_roster_df()


def _load_waiver_data() -> pd.DataFrame:
    """Load waiver wire rankings from MotherDuck with streak annotations."""
    try:
        with managed_connection() as conn:
            df: pd.DataFrame = conn.execute(f"""
                SELECT
                    ROW_NUMBER() OVER (ORDER BY w.overall_score DESC) AS rank,
                    p.player_id,
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
                    SELECT MAX(score_date) FROM {FACT_WAIVER_SCORES}
                    WHERE overall_score > 0
                )
                  AND w.overall_score > 0
                ORDER BY w.overall_score DESC
            """).fetchdf()

            if not df.empty:
                daily_df = _load_recent_daily_stats()
                if not daily_df.empty:
                    df = annotate_with_streaks(df, daily_df)
                elif "streak" not in df.columns:
                    df["streak"] = "—"
                return df
    except (duckdb.Error, Exception) as exc:
        logger.warning("Could not load waiver data from DB: %s", exc)
    return pd.DataFrame()


def _load_transactions() -> pd.DataFrame:
    """Load recent MLB transactions from MotherDuck."""
    try:
        with managed_connection() as conn:
            df: pd.DataFrame = conn.execute(f"""
                SELECT transaction_date, type, player_id, notes
                FROM {FACT_TRANSACTIONS}
                WHERE type IN (
                    'mlb_injury','mlb_activation','mlb_callup','mlb_demotion'
                )
                ORDER BY transaction_date DESC
                LIMIT 100
            """).fetchdf()
            if not df.empty:
                return df
    except (duckdb.Error, Exception) as exc:
        logger.warning("Could not load transactions from DB: %s", exc)
    return pd.DataFrame()


def _load_projections() -> pd.DataFrame:
    """Load the most recent season projections from MotherDuck.

    Returns one row per player (latest projection_date), falling back to an
    empty DataFrame when unavailable (projections are optional enrichment).
    """
    try:
        with managed_connection() as conn:
            df: pd.DataFrame = conn.execute(f"""
                SELECT
                    player_id, source,
                    proj_h, proj_hr, proj_sb, proj_bb,
                    proj_avg, proj_ops, proj_fpct,
                    proj_ip, proj_w, proj_k, proj_sv_h, proj_whip, proj_k_bb
                FROM {FACT_PROJECTIONS}
                WHERE projection_date = (
                    SELECT MAX(projection_date) FROM {FACT_PROJECTIONS}
                )
            """).fetchdf()
            return df
    except (duckdb.Error, Exception) as exc:
        logger.warning("Could not load projections from DB: %s", exc)
    return pd.DataFrame()


def _load_news(days: int = 3) -> pd.DataFrame:
    """Load recent player news from MotherDuck.

    Args:
        days: Number of days back to load news for.
    """
    try:
        with managed_connection() as conn:
            df: pd.DataFrame = conn.execute(f"""
                SELECT
                    id, player_id, player_name, headline, url,
                    source, published_at, sentiment_label, sentiment_score
                FROM {FACT_PLAYER_NEWS}
                WHERE fetched_at >= NOW() - INTERVAL '{days} days'
                ORDER BY published_at DESC
                LIMIT 200
            """).fetchdf()
            if not df.empty:
                return df
    except (duckdb.Error, Exception) as exc:
        logger.warning("Could not load news from DB: %s", exc)
    return pd.DataFrame()


# ── Server ─────────────────────────────────────────────────────────────────


def server(input: Inputs, output: Outputs, session: Session) -> None:
    """Shiny server function wiring reactive calcs and output renderers."""

    _refresh_counter: reactive.Value[int] = reactive.Value(0)

    # ── Reactive data ─────────────────────────────────────────────────────

    @reactive.calc
    def data_freshness() -> dict[str, object]:
        _refresh_counter()
        return _load_data_freshness()

    @render.ui
    def data_status_banner() -> Tag:
        freshness = data_freshness()
        is_offline = bool(freshness.get("is_offline", False))
        is_stale = bool(freshness.get("is_stale", False))
        generated_at = freshness.get("generated_at")
        report_date = freshness.get("report_date")
        if is_offline:
            error_detail = str(freshness.get("error", ""))
            msg = "⚠️ No pipeline data available — run the daily pipeline to populate."
            if error_detail:
                msg += f" ({error_detail[:120]})"
            return ui.div(ui.span(msg), class_="alert alert-danger mb-0 py-1")
        elif is_stale:
            return ui.div(
                ui.span(
                    f"⚠️ Showing most recent data (pipeline last ran: {report_date})"
                ),
                class_="alert alert-warning mb-0 py-1",
            )
        elif generated_at:
            return ui.div(
                ui.span(f"✅ Data as of {generated_at}"),
                class_="alert alert-success mb-0 py-1",
            )
        return ui.div()

    # Populate available weeks on startup and refresh
    @reactive.effect
    def _populate_week_choices() -> None:
        _refresh_counter()
        weeks = _load_available_weeks()
        ui.update_select("week_select", choices=weeks, selected="latest")

    @reactive.calc
    def selected_week() -> int | None:
        """Return the selected week number, or None for 'latest'."""
        val = str(input.week_select())
        if val == "latest":
            return None
        try:
            return int(val)
        except ValueError:
            return None

    @reactive.calc
    def daily_report() -> dict[str, Any]:
        _refresh_counter()
        return _load_daily_report(selected_week())

    @reactive.calc
    def matchup_data() -> pd.DataFrame:
        report = daily_report()
        rows = report.get("matchup_summary", [])
        if not isinstance(rows, list):
            return pd.DataFrame()
        return pd.DataFrame(rows)

    @reactive.calc
    def roster_data() -> pd.DataFrame:
        _refresh_counter()
        return _load_roster()

    @reactive.calc
    def waiver_data() -> pd.DataFrame:
        _refresh_counter()
        df = _load_waiver_data()
        pos = str(input.position_filter())
        if pos and pos != "All":
            df = df[df["position"].str.contains(pos, na=False)]
        if bool(input.callup_only()):
            df = df[df["callup"].astype(bool)]
        streak_filter = str(input.streak_filter())
        if streak_filter == "🔥 Hot":
            df = df[df["streak"] == "🔥 Hot"]
        elif streak_filter == "❄️ Cold":
            df = df[df["streak"] == "❄️ Cold"]
        return df.reset_index(drop=True)

    @reactive.calc
    def projection_data() -> pd.DataFrame:
        _refresh_counter()
        return _load_projections()

    @reactive.calc
    def transactions_data() -> pd.DataFrame:
        _refresh_counter()
        return _load_transactions()

    # ── Dashboard ─────────────────────────────────────────────────────────

    @render.ui
    def week_summary_ui() -> Tag:
        report = daily_report()
        report_date = str(report.get("report_date", "—"))
        week = report.get("week_number", "—")
        ip_pace = report.get("ip_pace", {})
        if isinstance(ip_pace, dict):
            current_ip = ip_pace.get("current_ip", 0.0)
            min_ip = ip_pace.get("min_ip", 21)
            projected_ip = ip_pace.get("projected_ip", 0.0)
            on_pace = ip_pace.get("on_pace", False)
        else:
            current_ip = min_ip = projected_ip = 0.0
            on_pace = False
        ip_icon = "✓" if on_pace else "⚠"
        ip_color = "#2e7d32" if on_pace else "#e65100"
        return ui.layout_columns(
            _stat_box("Report Date", str(report_date)),
            _stat_box("Matchup Week", f"Week {week}"),
            ui.tags.div(
                ui.tags.span(
                    "IP PACE",
                    style="font-size:0.65rem;font-weight:700;text-transform:uppercase;"
                    "letter-spacing:0.08em;color:#7ab8d4;display:block;",
                ),
                ui.tags.div(
                    ui.tags.span(f"{ip_icon} ", style=f"color:{ip_color};"),
                    ui.tags.span(
                        f"{current_ip}/{min_ip} IP",
                        style="font-weight:700;color:#fff;",
                    ),
                    ui.tags.span(
                        f"  proj {projected_ip}",
                        style="font-size:0.78rem;color:#8ab;",
                    ),
                    style="font-size:1.0rem;",
                ),
                style="background:#132747;border-radius:4px;border-left:3px solid "
                f"{'#1a7fa1' if on_pace else '#e65100'};"
                "padding:0.5rem 0.75rem;height:100%;",
            ),
            ui.div(
                ui.input_action_button(
                    "refresh_btn", "↻  Refresh", class_="btn-primary"
                ),
                class_="d-flex align-items-center justify-content-center",
            ),
            col_widths=[3, 3, 4, 2],
        )

    @render.ui
    def projected_wins_ui() -> Tag:
        df = matchup_data()
        if df.empty:
            return ui.p("No matchup data.", style="color:#888;padding:0.5rem;")
        probs = df["win_prob"].tolist()
        expected_wins = sum(probs)
        n = len(probs)
        mwp = match_win_probability(probs)
        projected_cat = round(expected_wins)
        if mwp >= 0.65:
            bar_color, outlook = "#2e7d32", "Favorable"
        elif mwp >= 0.40:
            bar_color, outlook = "#e65100", "Competitive"
        else:
            bar_color, outlook = "#c62828", "Difficult"
        safe_wins = int((df["win_prob"] >= 0.70).sum())
        in_play = int(((df["win_prob"] >= 0.35) & (df["win_prob"] < 0.70)).sum())
        safe_losses = int((df["win_prob"] < 0.35).sum())
        return ui.layout_columns(
            _stat_box("Projected Cats", f"{projected_cat}/{n}"),
            ui.tags.div(
                ui.tags.span(
                    "MATCH WIN PROB",
                    style="font-size:0.65rem;font-weight:700;text-transform:uppercase;"
                    "letter-spacing:0.08em;color:#7ab8d4;display:block;",
                ),
                ui.tags.div(
                    ui.tags.span(
                        f"{mwp * 100:.0f}%",
                        style=f"font-size:1.1rem;font-weight:700;color:{bar_color};",
                    ),
                    ui.tags.span(
                        f" {outlook}",
                        style=f"font-size:0.78rem;color:{bar_color};",
                    ),
                ),
                ui.tags.div(
                    ui.tags.div(
                        style=f"width:{mwp * 100:.0f}%;height:6px;"
                        f"background:{bar_color};border-radius:3px;",
                    ),
                    style="width:100%;background:#2a3f5a;border-radius:3px;margin-top:4px;",
                ),
                style="background:#132747;border-radius:4px;border-left:3px solid "
                f"{bar_color};padding:0.5rem 0.75rem;height:100%;",
            ),
            ui.tags.div(
                ui.tags.span(
                    "CATEGORY SPLIT",
                    style="font-size:0.65rem;font-weight:700;text-transform:uppercase;"
                    "letter-spacing:0.08em;color:#7ab8d4;display:block;",
                ),
                ui.tags.div(
                    ui.tags.span(
                        f"🟢 {safe_wins}",
                        style="color:#2e7d32;font-weight:700;margin-right:8px;",
                    ),
                    ui.tags.span(
                        f"🟡 {in_play}",
                        style="color:#e65100;font-weight:700;margin-right:8px;",
                    ),
                    ui.tags.span(
                        f"🔴 {safe_losses}",
                        style="color:#c62828;font-weight:700;",
                    ),
                    style="font-size:1.0rem;margin-top:2px;",
                ),
                style="background:#132747;border-radius:4px;border-left:3px solid "
                "#1a7fa1;padding:0.5rem 0.75rem;height:100%;",
            ),
            col_widths=[4, 4, 4],
        )

    @render.data_frame
    def lineup_table() -> pd.DataFrame:
        report = daily_report()
        lineup = report.get("lineup", {})
        if not isinstance(lineup, dict):
            return pd.DataFrame(columns=["Slot", "Player", "Streak"])
        roster = roster_data()
        id_to_name: dict[str, str] = {}
        id_to_streak: dict[str, str] = {}
        if not roster.empty:
            if "player_id" in roster.columns and "player_name" in roster.columns:
                id_to_name = dict(
                    zip(
                        roster["player_id"].astype(str),
                        roster["player_name"].astype(str),
                        strict=False,
                    )
                )
            if "player_id" in roster.columns and "streak" in roster.columns:
                id_to_streak = dict(
                    zip(
                        roster["player_id"].astype(str),
                        roster["streak"].astype(str),
                        strict=False,
                    )
                )
        rows = []
        for slot, pid in lineup.items():
            pid_str = str(pid)
            rows.append(
                {
                    "Slot": slot,
                    "Player": id_to_name.get(pid_str, pid_str),
                    "Streak": id_to_streak.get(pid_str, "—"),
                }
            )
        return pd.DataFrame(rows)

    @render.ui
    def matchup_scoreboard_ui() -> Tag:
        df = matchup_data()
        yahoo = _load_yahoo_matchup_stats(selected_week())
        if df.empty and not yahoo["mine"]:
            return ui.p(
                "No matchup data available.", style="padding:0.5rem;color:#666;"
            )
        headers = [
            "Category",
            "My Actual",
            "Opp Actual",
            "My Proj",
            "Opp Proj",
            "Win%",
            "Status",
        ]
        rows: list[list[Any]] = []
        for _, r in df.iterrows():
            status = str(r.get("status", ""))
            win_prob = float(r.get("win_prob", 0.5))
            cat = str(r.get("category", ""))
            cat_label = _CATEGORY_META.get(cat, {}).get("label", cat.upper())
            my_actual = yahoo["mine"].get(cat)
            opp_actual = yahoo["opp"].get(cat)
            rows.append(
                [
                    cat_label,
                    _fmt_stat(my_actual, cat),
                    _fmt_stat(opp_actual, cat),
                    _fmt_stat(r.get("my_value", ""), cat),
                    _fmt_stat(r.get("opp_value", ""), cat),
                    ui.tags.span(
                        f"{win_prob * 100:.0f}%",
                        class_=_win_pct_class(win_prob),
                    ),
                    ui.tags.span(
                        _STATUS_LABEL.get(status, status),
                        class_=_STATUS_CLASS.get(status, ""),
                    ),
                ]
            )
        return _html_table(headers, rows)

    @render.ui
    def adds_table() -> Tag:
        report = daily_report()
        raw_adds = report.get("adds", [])
        if not isinstance(raw_adds, list) or len(raw_adds) == 0:
            return ui.p(
                "No add recommendations today.", style="color:#888;padding:0.5rem;"
            )

        # Build ID→name lookup from roster (drops) and waiver wire (adds)
        id_to_name: dict[str, str] = {}
        roster = roster_data()
        if not roster.empty and "player_id" in roster.columns:
            for _, row in roster.iterrows():
                pid = str(row.get("player_id", ""))
                name = str(row.get("player_name", ""))
                if pid and name:
                    id_to_name[pid] = name
        waiver = waiver_data()
        if not waiver.empty and "player_id" in waiver.columns:
            for _, row in waiver.iterrows():
                pid = str(row.get("player_id", ""))
                name = str(row.get("player_name", ""))
                if pid and name:
                    id_to_name[pid] = name

        # Merge in live projections (keyed by player_id)
        proj_df = projection_data()
        proj_lookup: dict[str, dict[str, Any]] = {}
        if not proj_df.empty and "player_id" in proj_df.columns:
            for _, row in proj_df.iterrows():
                proj_lookup[str(row["player_id"])] = {
                    str(k): v for k, v in row.to_dict().items()
                }

        def _proj_row(
            player_id: str, override: dict[str, Any] | None
        ) -> dict[str, Any]:
            """Return projection dict: live DB > stub override > empty."""
            if player_id in proj_lookup:
                return proj_lookup[player_id]
            if override and isinstance(override, dict):
                return override
            return {}

        def _proj_stats_html(proj: dict[str, Any], is_pitcher: bool) -> Tag:
            """Render a compact row of projection stat boxes."""
            if not proj:
                return ui.tags.span(
                    "No projections available", style="color:#888;font-size:0.78rem;"
                )
            source = str(proj.get("source", "Projection"))
            boxes: list[Any] = []
            if is_pitcher:
                pairs = [
                    ("K", "proj_k"),
                    ("W", "proj_w"),
                    ("SV+H", "proj_sv_h"),
                    ("WHIP", "proj_whip"),
                    ("K/BB", "proj_k_bb"),
                ]
            else:
                pairs = [
                    ("AVG", "proj_avg"),
                    ("OPS", "proj_ops"),
                    ("HR", "proj_hr"),
                    ("SB", "proj_sb"),
                    ("H", "proj_h"),
                    ("BB", "proj_bb"),
                ]
            for label, key in pairs:
                raw = proj.get(key)
                if raw is None:
                    continue
                try:
                    fval = float(raw)
                except (TypeError, ValueError):
                    continue
                import math as _math

                if _math.isnan(fval):
                    continue
                # Extract category key from projection key (e.g. "proj_avg" → "avg")
                cat_key = key.removeprefix("proj_")
                display = _fmt_stat(fval, cat_key)
                boxes.append(
                    ui.tags.span(
                        ui.tags.span(
                            label,
                            style="font-size:0.6rem;color:#7ab8d4;font-weight:700;display:block;",
                        ),
                        ui.tags.span(
                            display,
                            style="font-size:0.88rem;font-weight:700;color:#fff;",
                        ),
                        style=(
                            "display:inline-flex;flex-direction:column;align-items:center;"
                            "background:#132747;border-radius:4px;padding:3px 8px;margin-right:5px;"
                        ),
                    )
                )
            if not boxes:
                return ui.tags.span(
                    "No projection data", style="color:#888;font-size:0.78rem;"
                )
            return ui.tags.div(
                ui.tags.span(
                    f"{source} projections: ",
                    style="font-size:0.7rem;color:#8ab;margin-right:4px;",
                ),
                *boxes,
                style="display:flex;flex-wrap:wrap;align-items:center;margin-top:4px;",
            )

        cards: list[Any] = []
        for item in raw_adds:
            a: dict[str, Any] = item if isinstance(item, dict) else {}
            add_id = str(a.get("add_player_id", ""))
            drop_id = str(a.get("drop_player_id", ""))
            add_name = id_to_name.get(add_id, add_id)
            drop_name = id_to_name.get(drop_id, drop_id)
            score = a.get("score", 0.0)
            cats = a.get("categories_improved", [])
            cat_labels = [
                _CATEGORY_META.get(c, {}).get("label", c.upper())
                for c in (cats if isinstance(cats, list) else [])
            ]
            add_pos = str(a.get("add_position", ""))
            drop_pos = str(a.get("drop_position", ""))
            add_streak = str(a.get("add_streak", "—"))
            drop_streak = str(a.get("drop_streak", "—"))
            callup_note = str(a.get("add_callup_note", ""))
            matchup_ctx = str(a.get("matchup_context", ""))

            # Determine pitcher vs batter for projection display
            pitcher_positions = {"SP", "RP", "P"}
            add_is_pitcher = bool(set(add_pos.split("/")) & pitcher_positions)
            drop_is_pitcher = bool(set(drop_pos.split("/")) & pitcher_positions)

            raw_add_proj = a.get("add_proj")
            add_proj = _proj_row(
                add_id, raw_add_proj if isinstance(raw_add_proj, dict) else None
            )
            raw_drop_proj = a.get("drop_proj")
            drop_proj = _proj_row(
                drop_id, raw_drop_proj if isinstance(raw_drop_proj, dict) else None
            )

            score_color = (
                "#2e7d32"
                if float(score) >= 7
                else "#e65100"
                if float(score) >= 5
                else "#888"
            )

            # Category pills
            cat_pills = [
                ui.tags.span(
                    lbl,
                    style=(
                        "background:#1a3a5c;color:#7ab8d4;font-size:0.68rem;font-weight:700;"
                        "padding:1px 7px;border-radius:8px;margin-right:3px;"
                    ),
                )
                for lbl in cat_labels
            ]

            # Player header row: ADD name + position + streak badge
            def _player_header(
                name: str, pos: str, streak: str, action: str, action_color: str
            ) -> Tag:
                return ui.tags.div(
                    ui.tags.span(
                        action,
                        style=f"font-size:0.65rem;font-weight:700;background:{action_color};"
                        "color:#fff;padding:1px 7px;border-radius:8px;margin-right:6px;"
                        "text-transform:uppercase;letter-spacing:0.06em;",
                    ),
                    ui.tags.span(
                        name,
                        style="font-size:1.0rem;font-weight:700;color:#fff;margin-right:6px;",
                    ),
                    ui.tags.span(
                        pos,
                        style="font-size:0.7rem;background:#1a3a5c;color:#7ab8d4;padding:1px 6px;"
                        "border-radius:6px;margin-right:6px;",
                    )
                    if pos
                    else ui.tags.span(),
                    _streak_badge(streak),
                    style="display:flex;align-items:center;flex-wrap:wrap;gap:2px;",
                )

            card = ui.tags.div(
                # Score badge
                ui.tags.div(
                    ui.tags.span(
                        f"Score: {float(score):.1f}",
                        style=f"font-size:0.72rem;font-weight:700;color:{score_color};",
                    ),
                    style="float:right;padding:0.25rem 0.5rem;",
                ),
                # Add row
                _player_header(add_name, add_pos, add_streak, "Add", "#2e7d32"),
                _proj_stats_html(add_proj, add_is_pitcher),
                # Drop row
                ui.tags.div(
                    ui.tags.span(
                        "↓",
                        style="font-size:0.9rem;color:#888;margin:6px 0 4px 0;display:block;",
                    ),
                ),
                _player_header(drop_name, drop_pos, drop_streak, "Drop", "#c62828"),
                _proj_stats_html(drop_proj, drop_is_pitcher),
                # Category pills + context
                ui.tags.div(
                    ui.tags.span(
                        "Improves: ",
                        style="font-size:0.7rem;color:#8ab;margin-right:4px;",
                    ),
                    *cat_pills,
                    style="margin-top:8px;display:flex;align-items:center;flex-wrap:wrap;",
                )
                if cat_pills
                else ui.tags.span(),
                ui.tags.div(
                    ui.tags.span(matchup_ctx, style="font-size:0.72rem;color:#a0b4c8;"),
                    style="margin-top:4px;",
                )
                if matchup_ctx
                else ui.tags.span(),
                ui.tags.div(
                    ui.tags.span(
                        "⬆ " + callup_note, style="font-size:0.72rem;color:#4fc3f7;"
                    ),
                    style="margin-top:2px;",
                )
                if callup_note
                else ui.tags.span(),
                style=(
                    "background:#0d1f38;border:1px solid #1e3a5f;border-radius:6px;"
                    "padding:0.75rem 1rem;margin-bottom:0.75rem;overflow:hidden;"
                ),
            )
            cards.append(card)

        return ui.tags.div(*cards, style="padding:0.25rem 0;")

    # ── Matchup Detail ────────────────────────────────────────────────────

    @render.ui
    def match_win_prob_ui() -> Tag:
        """Match win probability with scenario analysis."""
        df = matchup_data()
        if df.empty:
            return ui.p("No data.", style="color:#888;")

        probs = df["win_prob"].tolist()
        mwp = match_win_probability(probs)
        expected_wins = sum(probs)

        # Scenario analysis: best/worst case for contested categories
        contested = df[(df["win_prob"] >= 0.35) & (df["win_prob"] < 0.70)]
        best_probs = [0.95 if 0.35 <= p < 0.70 else p for p in probs]
        worst_probs = [0.05 if 0.35 <= p < 0.70 else p for p in probs]
        mwp_best = match_win_probability(best_probs)
        mwp_worst = match_win_probability(worst_probs)
        n_contested = len(contested)

        mwp_color = "#2e7d32" if mwp >= 0.55 else "#c62828"

        return ui.div(
            # Main probability display
            ui.tags.div(
                ui.tags.span(
                    f"{mwp * 100:.1f}%",
                    style=f"font-size:2rem;font-weight:800;color:{mwp_color};",
                ),
                ui.tags.span(
                    "  match win probability",
                    style="font-size:0.85rem;color:#4a6282;",
                ),
                style="margin-bottom:0.5rem;",
            ),
            # Expected wins
            ui.tags.div(
                ui.tags.span(
                    f"Expected category wins: {expected_wins:.1f} / 12",
                    style="font-size:0.85rem;color:#132747;font-weight:600;",
                ),
                style="margin-bottom:0.75rem;",
            ),
            # Scenario analysis
            ui.tags.p(
                "SCENARIO ANALYSIS",
                style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.07em;"
                "color:#4a6282;margin-bottom:6px;font-weight:700;",
            ),
            ui.tags.div(
                ui.tags.span(
                    f"{n_contested} contested categories in play",
                    style="font-size:0.8rem;color:#132747;margin-bottom:4px;display:block;",
                ),
                ui.tags.div(
                    _stat_box(
                        "Best Case",
                        f"{mwp_best * 100:.1f}%",
                        "win all toss-ups",
                        color="#2e7d32",
                    ),
                    _stat_box(
                        "Current",
                        f"{mwp * 100:.1f}%",
                        "projected",
                        color="#1a7fa1",
                    ),
                    _stat_box(
                        "Worst Case",
                        f"{mwp_worst * 100:.1f}%",
                        "lose all toss-ups",
                        color="#c62828",
                    ),
                    style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.5rem;"
                    "margin-top:0.5rem;",
                ),
            ),
            style="padding:0.25rem 0;",
        )

    @render.ui
    def bat_pitch_split_ui() -> Tag:
        """Batting vs pitching category record split."""
        df = matchup_data()
        if df.empty:
            return ui.p("No data.", style="color:#888;")

        bat_cats = [
            c
            for c in df["category"]
            if _CATEGORY_META.get(c, {}).get("type") == "batter"
        ]
        pitch_cats = [
            c
            for c in df["category"]
            if _CATEGORY_META.get(c, {}).get("type") == "pitcher"
        ]

        def _split_record(cats: list[str]) -> tuple[int, int, int]:
            wins = ties = losses = 0
            for cat in cats:
                row = df[df["category"] == cat]
                if row.empty:
                    continue
                wp = float(row.iloc[0]["win_prob"])
                if wp >= 0.55:
                    wins += 1
                elif wp <= 0.45:
                    losses += 1
                else:
                    ties += 1
            return wins, ties, losses

        bat_w, bat_t, bat_l = _split_record(bat_cats)
        pit_w, pit_t, pit_l = _split_record(pitch_cats)
        total_w = bat_w + pit_w
        total_l = bat_l + pit_l
        total_t = bat_t + pit_t

        def _record_bar(
            label: str, wins: int, ties: int, losses: int, total: int
        ) -> Tag:
            w_pct = (wins / total * 100) if total else 0
            t_pct = (ties / total * 100) if total else 0
            l_pct = (losses / total * 100) if total else 0
            return ui.tags.div(
                ui.tags.div(
                    ui.tags.span(
                        label,
                        style="font-size:0.75rem;font-weight:700;color:#132747;",
                    ),
                    ui.tags.span(
                        f" {wins}W - {ties}T - {losses}L",
                        style="font-size:0.75rem;color:#4a6282;",
                    ),
                    style="margin-bottom:3px;",
                ),
                ui.tags.div(
                    ui.tags.div(
                        style=f"width:{w_pct}%;background:#2e7d32;height:100%;border-radius:3px 0 0 3px;"
                        "display:inline-block;",
                    ),
                    ui.tags.div(
                        style=f"width:{t_pct}%;background:#e65100;height:100%;"
                        "display:inline-block;",
                    ),
                    ui.tags.div(
                        style=f"width:{l_pct}%;background:#c62828;height:100%;border-radius:0 3px 3px 0;"
                        "display:inline-block;",
                    ),
                    style="height:18px;background:#e0e0e0;border-radius:3px;overflow:hidden;"
                    "display:flex;",
                ),
                style="margin-bottom:0.75rem;",
            )

        # Identify which side needs focus
        bat_net = bat_w - bat_l
        pit_net = pit_w - pit_l
        if bat_net < pit_net:
            focus = "Batting categories need more attention this week."
            focus_color = "#c62828"
        elif pit_net < bat_net:
            focus = "Pitching categories need more attention this week."
            focus_color = "#c62828"
        else:
            focus = "Both sides are balanced."
            focus_color = "#2e7d32"

        return ui.div(
            _record_bar(
                "Overall", total_w, total_t, total_l, total_w + total_t + total_l
            ),
            _record_bar(
                f"Batting ({len(bat_cats)} cats)", bat_w, bat_t, bat_l, len(bat_cats)
            ),
            _record_bar(
                f"Pitching ({len(pitch_cats)} cats)",
                pit_w,
                pit_t,
                pit_l,
                len(pitch_cats),
            ),
            ui.tags.p(
                focus,
                style=f"font-size:0.78rem;color:{focus_color};font-weight:600;margin-top:0.25rem;",
            ),
            style="padding:0.25rem 0;",
        )

    @render.ui
    def battle_zones_ui() -> Tag:
        """Battle zone pills for all categories."""
        df = matchup_data()
        if df.empty:
            return ui.p("No data.", style="color:#888;")

        safe_wins_df = df[df["win_prob"] >= 0.70]
        in_play_df = df[(df["win_prob"] >= 0.35) & (df["win_prob"] < 0.70)]
        safe_losses_df = df[df["win_prob"] < 0.35]

        def _pills(sub: pd.DataFrame, color: str) -> list[Tag]:
            return [
                ui.tags.span(
                    f"{_CATEGORY_META.get(str(r['category']), {}).get('label', str(r['category']).upper())} "
                    f"{float(r['win_prob']) * 100:.0f}%",
                    style=f"display:inline-block;background:{color};color:#fff;"
                    "border-radius:4px;padding:2px 8px;margin:2px;"
                    "font-size:0.75rem;font-weight:600;",
                )
                for _, r in sub.iterrows()
            ]

        return ui.div(
            ui.tags.p(
                f"Safe Wins ({len(safe_wins_df)})",
                style="font-size:0.7rem;color:#2e7d32;font-weight:700;margin:0 0 2px 0;",
            ),
            ui.div(*_pills(safe_wins_df, "#2e7d32"))
            if not safe_wins_df.empty
            else ui.tags.span("None", style="color:#888;font-size:0.75rem;"),
            ui.tags.p(
                f"Contested ({len(in_play_df)})",
                style="font-size:0.7rem;color:#e65100;font-weight:700;margin:0.5rem 0 2px 0;",
            ),
            ui.div(*_pills(in_play_df, "#e65100"))
            if not in_play_df.empty
            else ui.tags.span("None", style="color:#888;font-size:0.75rem;"),
            ui.tags.p(
                f"Safe Losses ({len(safe_losses_df)})",
                style="font-size:0.7rem;color:#c62828;font-weight:700;margin:0.5rem 0 2px 0;",
            ),
            ui.div(*_pills(safe_losses_df, "#c62828"))
            if not safe_losses_df.empty
            else ui.tags.span("None", style="color:#888;font-size:0.75rem;"),
            style="padding:0.25rem 0;",
        )

    @render.ui
    def pitching_strategy_ui() -> Tag:
        """Pitching strategy: IP pace + category implications."""
        report = daily_report()
        df = matchup_data()
        ip_pace = report.get("ip_pace", {})
        if not isinstance(ip_pace, dict) or not ip_pace:
            return ui.p("No pitching data.", style="color:#888;")

        current_ip = float(ip_pace.get("current_ip", 0.0))
        projected_ip = float(ip_pace.get("projected_ip", 0.0))
        min_ip = int(ip_pace.get("min_ip", 21))
        on_pace = bool(ip_pace.get("on_pace", False))
        shortfall = max(0, min_ip - projected_ip)

        # Gather pitching category states
        pitch_cats = ["w", "k", "whip", "k_bb", "sv_h"]
        pitch_states: list[Tag] = []
        winning_pitch = 0
        losing_pitch = 0

        if not df.empty:
            for cat in pitch_cats:
                row = df[df["category"] == cat]
                if row.empty:
                    continue
                r = row.iloc[0]
                wp = float(r.get("win_prob", 0.5))
                my_val = r.get("my_value", 0)
                opp_val = r.get("opp_value", 0)
                cat_label = _CATEGORY_META.get(cat, {}).get("label", cat.upper())

                if wp >= 0.55:
                    winning_pitch += 1
                elif wp <= 0.45:
                    losing_pitch += 1

                color = (
                    "#2e7d32" if wp >= 0.55 else "#c62828" if wp <= 0.45 else "#e65100"
                )
                pitch_states.append(
                    ui.tags.div(
                        ui.tags.span(
                            cat_label,
                            style="font-weight:700;width:45px;display:inline-block;",
                        ),
                        ui.tags.span(
                            f"{_fmt_stat(my_val, cat)} vs {_fmt_stat(opp_val, cat)}",
                            style="font-size:0.78rem;color:#132747;",
                        ),
                        ui.tags.span(
                            f"  {wp * 100:.0f}%",
                            style=f"font-weight:700;color:{color};margin-left:6px;",
                        ),
                        style="font-size:0.8rem;margin-bottom:2px;",
                    )
                )

        # Build strategic advice
        advice_parts: list[Tag] = []
        if not on_pace:
            advice_parts.append(
                ui.tags.div(
                    ui.tags.span(
                        f"IP Shortfall: {shortfall:.1f} IP below minimum ({current_ip:.1f}/{min_ip})",
                        style="color:#c62828;font-weight:700;",
                    ),
                    ui.tags.span(
                        " — All 5 pitching categories forfeited if not met. "
                        "Stream pitchers to reach threshold.",
                        style="color:#4a6282;font-size:0.78rem;",
                    ),
                    style="font-size:0.8rem;margin-bottom:0.5rem;",
                )
            )
        else:
            # On pace — advise based on rate stats
            whip_row = df[df["category"] == "whip"] if not df.empty else pd.DataFrame()
            kbb_row = df[df["category"] == "k_bb"] if not df.empty else pd.DataFrame()

            whip_leading = (
                bool(whip_row.iloc[0].get("my_leads", False))
                if not whip_row.empty
                else False
            )
            kbb_leading = (
                bool(kbb_row.iloc[0].get("my_leads", False))
                if not kbb_row.empty
                else False
            )

            if whip_leading and kbb_leading:
                advice_parts.append(
                    ui.tags.div(
                        f"IP met ({current_ip:.1f}/{min_ip}). ",
                        ui.tags.span(
                            "Leading WHIP and K/BB — streaming risky pitchers could hurt rate stats. "
                            "Consider sitting marginal arms to protect your lead.",
                            style="color:#2e7d32;",
                        ),
                        style="font-size:0.8rem;margin-bottom:0.5rem;",
                    )
                )
            elif not whip_leading and not kbb_leading:
                advice_parts.append(
                    ui.tags.div(
                        f"IP met ({current_ip:.1f}/{min_ip}). ",
                        ui.tags.span(
                            "Trailing WHIP and K/BB — streaming high-K pitchers could flip both. "
                            "Target high-strikeout arms with manageable walk rates.",
                            style="color:#e65100;",
                        ),
                        style="font-size:0.8rem;margin-bottom:0.5rem;",
                    )
                )
            else:
                advice_parts.append(
                    ui.tags.div(
                        f"IP met ({current_ip:.1f}/{min_ip}). ",
                        ui.tags.span(
                            "Mixed rate stats — be selective with streaming. "
                            "Only add pitchers who help the category you're trailing.",
                            style="color:#4a6282;",
                        ),
                        style="font-size:0.8rem;margin-bottom:0.5rem;",
                    )
                )

        return ui.div(
            # Advice section
            *advice_parts,
            # Pitching category breakdown
            ui.tags.p(
                f"PITCHING CATEGORIES: {winning_pitch}W - {5 - winning_pitch - losing_pitch}T - {losing_pitch}L",
                style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.07em;"
                "color:#4a6282;margin:0.5rem 0 4px 0;font-weight:700;",
            ),
            *pitch_states,
            style="padding:0.25rem 0;",
        )

    @render.ui
    def category_gap_ui() -> Tag:
        """Category gap analysis — exact margins to flip contested categories."""
        df = matchup_data()
        if df.empty:
            return ui.p("No data.", style="color:#888;")

        contested = df[(df["win_prob"] >= 0.20) & (df["win_prob"] < 0.80)]
        if contested.empty:
            return ui.p(
                "No contested categories — all categories are decided.",
                style="color:#888;padding:0.5rem;",
            )

        headers = ["Category", "Mine", "Opp", "Gap", "Direction", "Difficulty"]
        rows: list[list[Any]] = []

        for _, r in contested.sort_values("win_prob", ascending=True).iterrows():
            cat = str(r.get("category", ""))
            cat_label = _CATEGORY_META.get(cat, {}).get("label", cat.upper())
            win_dir = _CATEGORY_META.get(cat, {}).get("win", "highest")
            my_val = float(r.get("my_value", 0))
            opp_val = float(r.get("opp_value", 0))
            my_leads = bool(r.get("my_leads", False))

            gap = abs(my_val - opp_val)

            # Format gap with units
            if cat in _INTEGER_CATS:
                gap_str = f"{int(round(gap))}"
            elif cat in _TWO_DECIMAL_CATS:
                gap_str = f"{gap:.2f}"
            else:
                gap_str = f"{gap:.3f}"

            # Direction text
            if my_leads:
                if win_dir == "lowest":
                    direction = f"Maintain {gap_str} lower"
                else:
                    direction = f"Maintain {gap_str} lead"
                dir_color = "#2e7d32"
            else:
                if win_dir == "lowest":
                    direction = f"Need to lower by {gap_str}"
                else:
                    direction = f"Need {gap_str} more"
                dir_color = "#c62828"

            # Difficulty rating based on gap relative to typical daily output
            margin_pct = float(r.get("margin_pct", 0))
            if margin_pct < 0.03:
                difficulty = ui.tags.span(
                    "Razor Thin",
                    style="background:#e65100;color:#fff;border-radius:3px;"
                    "padding:1px 6px;font-size:0.7rem;font-weight:600;",
                )
            elif margin_pct < 0.08:
                difficulty = ui.tags.span(
                    "Flippable",
                    style="background:#1a7fa1;color:#fff;border-radius:3px;"
                    "padding:1px 6px;font-size:0.7rem;font-weight:600;",
                )
            elif margin_pct < 0.15:
                difficulty = ui.tags.span(
                    "Reachable",
                    style="background:#558b2f;color:#fff;border-radius:3px;"
                    "padding:1px 6px;font-size:0.7rem;font-weight:600;",
                )
            else:
                difficulty = ui.tags.span(
                    "Tough",
                    style="background:#c62828;color:#fff;border-radius:3px;"
                    "padding:1px 6px;font-size:0.7rem;font-weight:600;",
                )

            rows.append(
                [
                    cat_label,
                    _fmt_stat(my_val, cat),
                    _fmt_stat(opp_val, cat),
                    gap_str,
                    ui.tags.span(
                        direction,
                        style=f"color:{dir_color};font-weight:600;font-size:0.8rem;",
                    ),
                    difficulty,
                ]
            )

        return _html_table(headers, rows)

    @render.ui
    def clutch_flip_ui() -> Tag:
        """Clutch flip analysis — marginal impact of flipping each category."""
        df = matchup_data()
        if df.empty:
            return ui.p("No data.", style="color:#888;")

        probs = df["win_prob"].tolist()
        mwp = match_win_probability(probs)

        in_play_df = df[(df["win_prob"] >= 0.35) & (df["win_prob"] < 0.70)]
        if in_play_df.empty:
            return ui.p(
                "No contested categories to flip.",
                style="color:#888;padding:0.5rem;",
            )

        # Build rows with delta for sorting
        flip_entries: list[tuple[float, list[Any]]] = []
        for _, r in in_play_df.iterrows():
            cat = str(r.get("category", ""))
            cat_label = _CATEGORY_META.get(cat, {}).get("label", cat.upper())
            wp = float(r.get("win_prob", 0.5))
            leading = bool(r.get("my_leads", False))

            # Marginal match win prob if this category flips
            probs_flipped = [1.0 - wp if p == wp else p for p in probs]
            mwp_flipped = match_win_probability(probs_flipped)
            delta = mwp_flipped - mwp
            delta_str = f"+{delta * 100:.1f}%" if delta >= 0 else f"{delta * 100:.1f}%"

            # Priority ranking
            if delta >= 0.10:
                priority = ui.tags.span(
                    "High",
                    style="background:#2e7d32;color:#fff;border-radius:3px;"
                    "padding:1px 6px;font-size:0.7rem;font-weight:600;",
                )
            elif delta >= 0.04:
                priority = ui.tags.span(
                    "Medium",
                    style="background:#e65100;color:#fff;border-radius:3px;"
                    "padding:1px 6px;font-size:0.7rem;font-weight:600;",
                )
            else:
                priority = ui.tags.span(
                    "Low",
                    style="background:#888;color:#fff;border-radius:3px;"
                    "padding:1px 6px;font-size:0.7rem;font-weight:600;",
                )

            flip_entries.append(
                (
                    abs(delta),
                    [
                        cat_label,
                        f"{'▲' if leading else '▼'} {_fmt_stat(r.get('my_value', ''), cat)}",
                        _fmt_stat(r.get("opp_value", ""), cat),
                        ui.tags.span(f"{wp * 100:.0f}%", class_=_win_pct_class(wp)),
                        ui.tags.span(
                            delta_str,
                            style="font-weight:700;color:"
                            + ("#1a7fa1" if delta >= 0 else "#c62828")
                            + ";",
                        ),
                        priority,
                    ],
                )
            )

        # Sort by absolute delta descending (highest impact first)
        flip_entries.sort(key=lambda x: x[0], reverse=True)
        flip_rows = [row for _, row in flip_entries]

        return _html_table(
            ["Category", "Mine", "Opp", "Win%", "Match Win% Delta", "Priority"],
            flip_rows,
        )

    # ── Waiver Wire ───────────────────────────────────────────────────────

    @render.ui
    def waiver_table_ui() -> Tag:
        df = waiver_data()
        if df.empty:
            return ui.p("No waiver wire data.", style="color:#888;padding:0.5rem;")
        headers = ["#", "Player", "Team", "Pos", "Score ⓘ", "Streak", "Key Stats"]
        rows_out: list[list[Any]] = []
        for _, r in df.iterrows():
            pos = str(r.get("position", ""))
            is_p = any(p in pos.upper() for p in ("SP", "RP"))
            if is_p:
                key_stats = (
                    f"W:{r.get('w', '—')} K:{r.get('k', '—')} "
                    f"WHIP:{r.get('whip', '—')} SV+H:{r.get('sv_h', '—')}"
                )
            else:
                key_stats = (
                    f"AVG:{r.get('avg', '—')} OPS:{r.get('ops', '—')} "
                    f"HR:{r.get('hr', '—')} SB:{r.get('sb', '—')}"
                )
            callup_badge = (
                ui.tags.span(
                    " ⬆",
                    style="background:#f5a623;color:#132747;border-radius:3px;"
                    "padding:1px 4px;font-size:0.68rem;font-weight:700;margin-left:4px;",
                )
                if r.get("callup")
                else ui.tags.span()
            )
            rows_out.append(
                [
                    str(int(r.get("rank", 0))),
                    ui.tags.span(str(r.get("player_name", "")), callup_badge),
                    str(r.get("team", "")),
                    pos,
                    ui.tags.span(
                        f"{float(r.get('score', 0)):.1f}",
                        style="font-weight:700;color:#132747;",
                    ),
                    _streak_badge(str(r.get("streak", "—"))),
                    ui.tags.span(key_stats, style="font-size:0.73rem;color:#4a6282;"),
                ]
            )
        return _html_table(headers, rows_out)

    # ── Roster ────────────────────────────────────────────────────────────

    @render.ui
    def hitter_roster_ui() -> Tag:
        df = roster_data()
        if df.empty:
            return ui.p("No roster data.", style="color:#888;")
        hitters = df[~df["slot"].isin(_PITCHER_SLOTS)].copy()
        if hitters.empty:
            return ui.p("No hitters found.", style="color:#888;")
        headers = [
            "Slot",
            "Player",
            "Pos",
            "H",
            "HR",
            "SB",
            "BB",
            "AVG",
            "OPS",
            "Streak",
        ]
        rows_out: list[list[Any]] = []
        for _, r in hitters.iterrows():
            rows_out.append(
                [
                    str(r.get("slot", "")),
                    str(r.get("player_name", "")),
                    str(r.get("position", "")),
                    _fmt_stat(r.get("h"), "h"),
                    _fmt_stat(r.get("hr"), "hr"),
                    _fmt_stat(r.get("sb"), "sb"),
                    _fmt_stat(r.get("bb"), "bb"),
                    _fmt_stat(r.get("avg"), "avg"),
                    _fmt_stat(r.get("ops"), "ops"),
                    _streak_badge(str(r.get("streak", "—"))),
                ]
            )
        return _html_table(headers, rows_out)

    @render.ui
    def pitcher_roster_ui() -> Tag:
        df = roster_data()
        if df.empty:
            return ui.p("No roster data.", style="color:#888;")
        pitchers = df[df["slot"].isin(_PITCHER_SLOTS)].copy()
        if pitchers.empty:
            return ui.p("No pitchers found.", style="color:#888;")
        headers = ["Slot", "Player", "Pos", "W", "K", "WHIP", "K/BB", "SV+H", "Streak"]
        rows_out: list[list[Any]] = []
        for _, r in pitchers.iterrows():
            whip_val = float(r.get("whip", 0.0))
            whip_color = (
                "#2e7d32"
                if whip_val < 1.00
                else "#e65100"
                if whip_val < 1.30
                else "#c62828"
            )
            rows_out.append(
                [
                    str(r.get("slot", "")),
                    str(r.get("player_name", "")),
                    str(r.get("position", "")),
                    _fmt_stat(r.get("w", "—"), "w"),
                    _fmt_stat(r.get("k", "—"), "k"),
                    ui.tags.span(
                        _fmt_stat(whip_val, "whip") if whip_val else "—",
                        style=f"color:{whip_color};font-weight:600;",
                    ),
                    _fmt_stat(r.get("k_bb", 0.0), "k_bb"),
                    _fmt_stat(r.get("sv_h", "—"), "sv_h"),
                    _streak_badge(str(r.get("streak", "—"))),
                ]
            )
        return _html_table(headers, rows_out)

    # ── Transactions ──────────────────────────────────────────────────────

    @render.ui
    def transactions_ui() -> Tag:
        df = transactions_data()
        type_filter = str(input.transaction_type_filter())
        if type_filter != "All":
            df = df[df["type"] == type_filter]
        if df.empty:
            return ui.p("No transactions found.", style="color:#888;padding:0.5rem;")
        headers = ["Date", "Type", "Player", "Notes"]
        rows_out: list[list[Any]] = []
        for _, r in df.head(50).iterrows():
            t = str(r.get("type", ""))
            label, color = _TRANSACTION_TYPE_LABELS.get(t, (t, "#333"))
            player_name = str(r.get("player_name", r.get("player_id", "—")))
            team = str(r.get("team", ""))
            pos = str(r.get("position", ""))
            rows_out.append(
                [
                    str(r.get("transaction_date", "—"))[:10],
                    ui.tags.span(
                        label,
                        style=f"color:{color};font-weight:700;font-size:0.75rem;",
                    ),
                    ui.tags.span(
                        player_name,
                        ui.tags.span(
                            f" {team} · {pos}" if team and pos else "",
                            style="color:#4a6282;font-size:0.75rem;",
                        ),
                    ),
                    ui.tags.span(
                        str(r.get("notes", "")),
                        style="font-size:0.75rem;color:#4a6282;",
                    ),
                ]
            )
        return _html_table(headers, rows_out)

    # ── Trades ────────────────────────────────────────────────────────────

    @render.ui
    def trades_ui() -> Tag:
        report = daily_report()
        trades = report.get("trades", [])
        if not trades:
            return ui.p("No trade proposals available.", style="color:#888;")
        cards: list[Any] = []
        for trade in trades:
            acc_raw = trade.get("acceptance_pct", 50)
            acc = int(acc_raw) if isinstance(acc_raw, (int, float)) else 50
            bar_color = (
                "#2e7d32" if acc >= 65 else "#e65100" if acc >= 50 else "#c62828"
            )
            give = str(trade.get("give_player", ""))
            give_team = str(trade.get("give_team", ""))
            give_helps = trade.get("give_helps", [])
            receive = str(trade.get("receive_player", ""))
            receive_team = str(trade.get("receive_team", ""))
            receive_helps = trade.get("receive_helps", [])
            gain = str(trade.get("my_category_gain", ""))
            rationale = str(trade.get("rationale", ""))

            def _pills(cats: object, bg: str) -> list[Tag]:
                if not isinstance(cats, list):
                    return []
                return [
                    ui.tags.span(
                        str(c),
                        style=f"background:{bg};color:#fff;border-radius:3px;"
                        "padding:1px 6px;margin:1px;font-size:0.68rem;font-weight:600;",
                    )
                    for c in cats
                ]

            cards.append(
                ui.div(
                    ui.layout_columns(
                        ui.div(
                            ui.tags.div(
                                ui.tags.span(
                                    "YOU GIVE",
                                    style="font-size:0.65rem;font-weight:700;"
                                    "text-transform:uppercase;color:#c62828;display:block;",
                                ),
                                ui.tags.div(
                                    ui.tags.strong(give),
                                    ui.tags.span(
                                        f" ({give_team})",
                                        style="color:#4a6282;font-size:0.8rem;",
                                    ),
                                    style="font-size:0.95rem;margin:2px 0;",
                                ),
                                ui.div(
                                    *_pills(give_helps, "#4a6282"),
                                    style="margin-top:2px;",
                                ),
                                style="padding:0.5rem;",
                            ),
                        ),
                        ui.div(
                            ui.tags.span(
                                "⇌",
                                style="font-size:1.4rem;color:#1a7fa1;",
                            ),
                            class_="d-flex align-items-center justify-content-center",
                        ),
                        ui.div(
                            ui.tags.div(
                                ui.tags.span(
                                    "YOU GET",
                                    style="font-size:0.65rem;font-weight:700;"
                                    "text-transform:uppercase;color:#2e7d32;display:block;",
                                ),
                                ui.tags.div(
                                    ui.tags.strong(receive),
                                    ui.tags.span(
                                        f" ({receive_team})",
                                        style="color:#4a6282;font-size:0.8rem;",
                                    ),
                                    style="font-size:0.95rem;margin:2px 0;",
                                ),
                                ui.div(
                                    *_pills(receive_helps, "#1a7fa1"),
                                    style="margin-top:2px;",
                                ),
                                style="padding:0.5rem;",
                            ),
                        ),
                        col_widths=[5, 2, 5],
                    ),
                    ui.div(
                        ui.tags.div(
                            ui.tags.span(
                                "NET GAIN: ",
                                style="font-weight:700;color:#132747;font-size:0.75rem;",
                            ),
                            ui.tags.span(
                                gain, style="color:#1a7fa1;font-size:0.75rem;"
                            ),
                            style="margin-bottom:4px;",
                        ),
                        ui.tags.div(
                            rationale,
                            style="font-size:0.77rem;color:#4a6282;margin-bottom:6px;",
                        ),
                        ui.layout_columns(
                            ui.div(
                                ui.tags.span(
                                    "EST. ACCEPTANCE: ",
                                    style="font-size:0.72rem;font-weight:700;color:#132747;",
                                ),
                                ui.tags.span(
                                    f"{acc}%",
                                    style=f"font-size:0.72rem;font-weight:700;color:{bar_color};",
                                ),
                                ui.div(
                                    ui.div(
                                        style=f"width:{acc}%;height:5px;"
                                        f"background:{bar_color};border-radius:3px;",
                                    ),
                                    style="width:100%;background:#dde3eb;"
                                    "border-radius:3px;margin-top:2px;",
                                ),
                            ),
                            col_widths=[12],
                        ),
                        style="padding:0.4rem 0.75rem 0.5rem 0.75rem;"
                        "border-top:1px solid #edf1f7;",
                    ),
                    style="border:1px solid #d8e1eb;border-radius:6px;"
                    "margin-bottom:0.75rem;background:#fff;",
                )
            )
        return ui.div(*cards)

    # ── News ──────────────────────────────────────────────────────────────

    @reactive.calc
    def news_data() -> pd.DataFrame:
        _refresh_counter()
        return _load_news()

    @reactive.effect
    def _populate_news_player_filter() -> None:
        """Populate the player dropdown with names from loaded news."""
        df = news_data()
        names = (
            sorted(df["player_name"].dropna().unique().tolist()) if not df.empty else []
        )
        choices: dict[str, str] = {"All": "All Players"}
        choices.update({n: n for n in names})
        ui.update_select("news_player_filter", choices=choices)

    @render.ui
    def news_ui() -> Tag:
        df = news_data()

        sentiment_filter = str(input.news_sentiment_filter())
        player_filter = str(input.news_player_filter())

        if not df.empty:
            if sentiment_filter != "All":
                df = df[df["sentiment_label"] == sentiment_filter]
            if player_filter != "All":
                df = df[df["player_name"] == player_filter]

        if df.empty:
            return ui.div("No news available.", style="color:#8096b0;padding:1rem;")

        _sentiment_style: dict[str, tuple[str, str]] = {
            "Good": ("#2e7d32", "✅ Good"),
            "Bad": ("#c62828", "🚨 Bad"),
            "Informative": ("#1a7fa1", "ℹ️ Informative"),
        }

        cards: list[Any] = []
        for _, row in df.iterrows():
            label = str(row.get("sentiment_label", "Informative"))
            color, badge_text = _sentiment_style.get(label, ("#8096b0", label))
            player = str(row.get("player_name", ""))
            headline = str(row.get("headline", ""))
            source = str(row.get("source", ""))
            url = str(row.get("url", ""))
            pub = row.get("published_at", "")
            pub_str = str(pub)[:10] if pub else ""

            headline_el: Any = (
                ui.tags.a(
                    headline,
                    href=url,
                    target="_blank",
                    rel="noopener noreferrer",
                    style="color:#1a7fa1;text-decoration:underline;font-weight:600;"
                    "font-size:0.88rem;",
                )
                if url
                else ui.tags.span(
                    headline,
                    style="color:#132747;font-weight:600;font-size:0.88rem;",
                )
            )

            cards.append(
                ui.div(
                    ui.div(
                        ui.tags.span(
                            badge_text,
                            style=f"background:{color};color:#fff;padding:2px 8px;"
                            "border-radius:10px;font-size:0.72rem;font-weight:700;"
                            "margin-right:0.5rem;",
                        ),
                        ui.tags.span(
                            player,
                            style="font-size:0.78rem;font-weight:700;color:#1a7fa1;"
                            "margin-right:0.5rem;",
                        ),
                        ui.tags.span(
                            f"{source}  ·  {pub_str}" if source else pub_str,
                            style="font-size:0.73rem;color:#8096b0;",
                        ),
                        style="margin-bottom:4px;",
                    ),
                    headline_el,
                    style="padding:0.55rem 0.85rem;border-top:1px solid #edf1f7;"
                    "background:#fff;",
                )
            )

        return ui.div(
            *cards,
            style="border:1px solid #d8e1eb;border-radius:6px;overflow:hidden;",
        )

    # ── Manual refresh ────────────────────────────────────────────────────

    @reactive.effect
    @reactive.event(input.refresh_btn)
    def _refresh() -> None:
        _refresh_counter.set(_refresh_counter() + 1)
