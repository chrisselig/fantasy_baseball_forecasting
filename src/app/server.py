"""
server.py

Reactive server logic for the Shiny app.
All expensive operations go in @reactive.calc (cached).
Returns empty results when data is unavailable.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import duckdb
import pandas as pd
from htmltools import Tag
from shiny import Inputs, Outputs, Session, reactive, render, ui

from src.analysis.hot_cold import annotate_with_streaks, match_win_probability
from src.analysis.lineup_optimizer import _position_is_pitcher
from src.config import load_league_settings
from src.db.connection import managed_connection
from src.db.schema import (
    DIM_PLAYERS,
    FACT_DAILY_REPORTS,
    FACT_MATCHUPS,
    FACT_PLAYER_ADVANCED_STATS,
    FACT_PLAYER_NEWS,
    FACT_PLAYER_STATS_DAILY,
    FACT_PROJECTIONS,
    FACT_ROSTERS,
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


# Tooltip text for every roster column. Each entry is a single string with
# embedded newlines so it renders as a multi-line native browser tooltip.
_ROSTER_TOOLTIPS: dict[str, str] = {
    # Hitter counting / rate stats
    "H": "Hits — week to date.\nGood week: 10+  •  Avg: 6-9  •  Poor: <5",
    "HR": "Home runs — week to date.\nGood week: 2+  •  Avg: 1  •  Poor: 0",
    "SB": "Stolen bases — week to date.\nGood week: 2+  •  Avg: 1  •  Poor: 0",
    "BB": "Walks — week to date.\nGood week: 5+  •  Avg: 3-4  •  Poor: 0-2",
    "AVG": (
        "Batting Average — H / AB (week to date).\n"
        "Calc: hits divided by at-bats.\n"
        "Elite: .300+  •  Good: .270-.299  •  Avg: .240-.269  •  Poor: <.240"
    ),
    "OPS": (
        "On-base + Slugging — week to date.\n"
        "Calc: OBP + SLG.\n"
        "Elite: .900+  •  Good: .800-.899  •  Avg: .700-.799  •  Poor: <.700"
    ),
    # Hitter advanced
    "wOBA": (
        "Weighted On-Base Average — season to date (computed from raw stats).\n"
        "Calc: linear weights × singles, doubles, triples, HR, BB, HBP, "
        "divided by PA. One number that captures all offensive value.\n"
        "Elite: .400+  •  Good: .360-.399  •  Avg: .320-.359  •  Poor: <.310"
    ),
    "xwOBA": (
        "Expected wOBA — what wOBA *should* be based on Statcast batted-ball "
        "quality (exit velocity, launch angle), removing luck and defense.\n"
        "If xwOBA > wOBA the player has been unlucky and is due to improve.\n"
        "Elite: .380+  •  Good: .340-.379  •  Avg: .310-.339  •  Poor: <.300"
    ),
    "Barrel%": (
        "Barrel rate — % of batted balls hit in the optimal exit-velocity / "
        "launch-angle combo (~98+ mph at 26-30 deg). Best predictor of HR power.\n"
        "Elite: 12%+  •  Good: 8-11%  •  Avg: 5-7%  •  Poor: <5%"
    ),
    "HardHit%": (
        "Hard-hit rate — % of batted balls with exit velocity ≥ 95 mph.\n"
        "Elite: 50%+  •  Good: 42-49%  •  Avg: 35-41%  •  Poor: <35%"
    ),
    "LA": (
        "Avg Launch Angle (degrees) — vertical angle off the bat.\n"
        "Sweet zone for damage: 8-32 degrees. Below 8 = grounders, "
        "above 32 = pop-ups.\n"
        "Power hitter: 15-22  •  Line-drive bat: 8-14  •  Ground-ball: <8"
    ),
    "SwSp%": (
        "Sweet-Spot % — % of batted balls in the 8-32° launch-angle window.\n"
        "Elite: 38%+  •  Good: 34-37%  •  Avg: 30-33%  •  Poor: <30%"
    ),
    "BatSp": (
        "Bat Speed — Statcast percentile (0-100) for average swing speed.\n"
        "Higher = more raw power potential. Available 2024+.\n"
        "Elite: 90+  •  Good: 70-89  •  Avg: 40-69  •  Poor: <40"
    ),
    # Pitcher counting / rate
    "W": "Wins — week to date.\nGood week: 2+  •  Avg: 1  •  Poor: 0",
    "K": "Strikeouts — week to date.\nGood week: 12+  •  Avg: 7-11  •  Poor: <6",
    "WHIP": (
        "Walks + Hits per Inning Pitched (week to date). LOWEST WINS in this "
        "league.\nCalc: (BB + H) / IP.\n"
        "Elite: <1.00  •  Good: 1.00-1.20  •  Avg: 1.21-1.35  •  Poor: >1.35"
    ),
    "K/BB": (
        "Strikeout-to-Walk ratio — week to date.\n"
        "Calc: K / BB.\n"
        "Elite: 5.0+  •  Good: 3.5-4.9  •  Avg: 2.5-3.4  •  Poor: <2.5"
    ),
    "SV+H": (
        "Saves + Holds — week to date. Combined relief category.\n"
        "Good week (closer): 3+  •  Avg: 1-2  •  Poor: 0"
    ),
    # Pitcher advanced
    "xERA": (
        "Expected ERA — Statcast estimate of ERA based on quality of contact "
        "allowed (exit velocity, launch angle), removing luck/defense.\n"
        "If xERA > ERA the pitcher has been lucky and is due to regress.\n"
        "Elite: <3.00  •  Good: 3.00-3.75  •  Avg: 3.76-4.50  •  Poor: >4.50"
    ),
    "xwOBA-A": (
        "Expected wOBA Against — what opponents' wOBA *should* be based on "
        "their Statcast contact quality off this pitcher.\n"
        "Elite: <.290  •  Good: .290-.310  •  Avg: .311-.330  •  Poor: >.330"
    ),
    "K-BB%": (
        "Strikeout minus Walk rate — season to date (computed).\n"
        "Calc: (K − BB) / batters faced × 100. Best single-stat measure of "
        "pitcher dominance.\n"
        "Elite: 20%+  •  Good: 15-19%  •  Avg: 10-14%  •  Poor: <10%"
    ),
    "Brl%-A": (
        "Barrel % Against — rate of barrels (the most dangerous batted-ball "
        "type) allowed by this pitcher. Lower is better.\n"
        "Elite: <5%  •  Good: 5-7%  •  Avg: 8-10%  •  Poor: >10%"
    ),
    # Misc
    "Slot": "Roster slot assigned by Yahoo (active position, BN, IL, NA).",
    "Player": "Player name.",
    "Pos": "Eligible positions.",
    "Streak": (
        "🔥 Hot or ❄️ Cold tag from last 7 days (hitters) / 10 days (pitchers).\n"
        "Hitter hot: 2+ of (hit streak, AVG ≥ .320, OPS ≥ .920, recent HR/SB).\n"
        "Pitcher hot: 2+ of (WHIP < 1.00, RA9 < 2.50, K/9 > 9.0, K/BB > 3.0)."
    ),
}


def _fmt_adv(value: Any, digits: int = 3) -> str:
    """Format an advanced-stat value with a decimal style, or '—' if NaN/None."""
    try:
        if value is None:
            return "—"
        f = float(value)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(f):
        return "—"
    return f"{f:.{digits}f}"


# ── Roster stat tier colouring ────────────────────────────────────────────
#
# Each roster stat has tooltip-documented thresholds for Elite / Great /
# Average / Poor. ``_STAT_TIER_FNS`` maps a stat key to a function that
# returns the tier name for a given numeric value. Counting stats that have
# only three documented buckets (H, HR, SB, BB, W, K, SV+H) skip the Elite
# tier by passing ``None`` as the elite cutoff.

_TIER_COLORS: dict[str, str] = {
    "elite": "#1b5e20",  # dark green
    "great": "#1565c0",  # blue
    "average": "#b8860b",  # amber
    "poor": "#c62828",  # red
}

_TIER_ORDER: tuple[str, ...] = ("elite", "great", "average", "poor")


def _tier_higher(v: float, elite: float | None, great: float, average: float) -> str:
    """Tier for a higher-is-better stat. ``elite=None`` skips the elite tier."""
    if elite is not None and v >= elite:
        return "elite"
    if v >= great:
        return "great"
    if v >= average:
        return "average"
    return "poor"


def _tier_lower(v: float, elite: float, great: float, average: float) -> str:
    """Tier for a lower-is-better stat (WHIP, xERA, xwOBA-A, Brl%-A)."""
    if v < elite:
        return "elite"
    if v <= great:
        return "great"
    if v <= average:
        return "average"
    return "poor"


def _tier_launch_angle(v: float) -> str:
    """Tier for Avg Launch Angle — a range-type stat (no elite tier).

    Power zone 15–22, line-drive zone 8–14 or 23–32, else poor.
    """
    if 15 <= v <= 22:
        return "great"
    if 8 <= v < 15 or 22 < v <= 32:
        return "average"
    return "poor"


_STAT_TIER_FNS: dict[str, Callable[[float], str]] = {
    # Hitter counting — 3-tier (no elite)
    "h": lambda v: _tier_higher(v, None, 10, 6),
    "hr": lambda v: _tier_higher(v, None, 2, 1),
    "sb": lambda v: _tier_higher(v, None, 2, 1),
    "bb": lambda v: _tier_higher(v, None, 5, 3),
    # Hitter rate — 4-tier
    "avg": lambda v: _tier_higher(v, 0.300, 0.270, 0.240),
    "ops": lambda v: _tier_higher(v, 0.900, 0.800, 0.700),
    "woba": lambda v: _tier_higher(v, 0.400, 0.360, 0.320),
    "xwoba": lambda v: _tier_higher(v, 0.380, 0.340, 0.310),
    "barrel_pct": lambda v: _tier_higher(v, 12, 8, 5),
    "hard_hit_pct": lambda v: _tier_higher(v, 50, 42, 35),
    "sweet_spot_pct": lambda v: _tier_higher(v, 38, 34, 30),
    "bat_speed_pctile": lambda v: _tier_higher(v, 90, 70, 40),
    "avg_launch_angle": _tier_launch_angle,
    # Pitcher counting — 3-tier
    "w": lambda v: _tier_higher(v, None, 2, 1),
    "k": lambda v: _tier_higher(v, None, 12, 7),
    "sv_h": lambda v: _tier_higher(v, None, 3, 1),
    # Pitcher rate — lower is better
    "whip": lambda v: _tier_lower(v, 1.00, 1.20, 1.35),
    "xera": lambda v: _tier_lower(v, 3.00, 3.75, 4.50),
    "xwoba_against": lambda v: _tier_lower(v, 0.290, 0.310, 0.330),
    "barrel_pct_against": lambda v: _tier_lower(v, 5, 7, 10),
    # Pitcher rate — higher is better
    "k_bb": lambda v: _tier_higher(v, 5.0, 3.5, 2.5),
    "k_bb_pct": lambda v: _tier_higher(v, 20, 15, 10),
}


def _tier_for(stat_key: str, value: Any) -> str | None:
    """Return the tier name for ``value`` under ``stat_key`` (or None)."""
    fn = _STAT_TIER_FNS.get(stat_key)
    if fn is None or value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(v):
        return None
    return fn(v)


def _color_tier(display: str, tier: str | None) -> Any:
    """Wrap ``display`` in a coloured span for its tier, or pass through."""
    if tier is None or display in ("", "—"):
        return display
    color = _TIER_COLORS.get(tier)
    if color is None:
        return display
    return ui.tags.span(display, style=f"color:{color};font-weight:700;")


def _fmt_stat_tier(value: Any, stat_key: str, *, per_game: bool = False) -> Any:
    """``_fmt_stat`` wrapped in tier colour based on ``stat_key``."""
    return _color_tier(
        _fmt_stat(value, stat_key, per_game=per_game),
        _tier_for(stat_key, value),
    )


def _fmt_adv_tier(value: Any, stat_key: str, digits: int) -> Any:
    """``_fmt_adv`` wrapped in tier colour based on ``stat_key``."""
    return _color_tier(_fmt_adv(value, digits), _tier_for(stat_key, value))


def _roster_tier_legend() -> Tag:
    """Compact horizontal legend: Elite / Great / Average / Poor swatches."""
    swatch_style = (
        "display:inline-block;width:12px;height:12px;border-radius:2px;"
        "margin-right:4px;vertical-align:middle;"
    )
    item_style = (
        "display:inline-flex;align-items:center;margin-right:14px;"
        "font-size:0.72rem;color:#556;font-weight:600;"
    )
    items: list[Any] = []
    for tier in _TIER_ORDER:
        items.append(
            ui.tags.span(
                ui.tags.span(style=f"{swatch_style}background:{_TIER_COLORS[tier]};"),
                tier.capitalize(),
                style=item_style,
            )
        )
    return ui.div(*items, style="margin:0.15rem 0 0.4rem 0;")


def _roster_slot_order(slot: Any) -> int:
    """Sort key so active slots appear first, then BN, then IL/NA."""
    s = str(slot).strip().upper()
    if s in {"BN", "BENCH"}:
        return 1
    if s.startswith("IL"):
        return 2
    if s in {"NA", "N/A"}:
        return 3
    return 0


_TRANSACTION_TYPE_LABELS: dict[str, tuple[str, str]] = {
    "call_up": ("⬆ Call-Up", "#2e7d32"),
    "il_activation": ("✅ IL Activated", "#1a7fa1"),
    "il_placement": ("🏥 Placed on IL", "#e65100"),
    "demotion": ("⬇ Demotion", "#c62828"),
    "status_change": ("🔄 Status Change", "#555555"),
    "dfa": ("✂️ DFA", "#6a1b9a"),
    "released": ("🚫 Released", "#333333"),
}


def _win_pct_class(pct: float) -> str:
    """CSS class for win probability colouring."""
    if pct >= 0.65:
        return "win-high"
    if pct >= 0.35:
        return "win-mid"
    return "win-low"


def _th_tip(label: str, tip: str) -> Tag:
    """Header cell with a native HTML tooltip (cursor changes on hover)."""
    return ui.tags.th(
        label,
        title=tip,
        style="cursor:help;border-bottom:1px dotted #6b8aa8;",
    )


def _html_table(
    headers: list[Any],
    rows: list[list[Any]],
    *,
    group_headers: list[tuple[str, int]] | None = None,
) -> Tag:
    """Build a styled HTML table matching the Savant theme.

    Headers may be plain strings or pre-built ``ui.tags.th(...)`` Tags
    (used by ``_th_tip`` to attach tooltip metadata).

    ``group_headers`` optionally renders a second header row *above* the
    column headers, where each entry is ``(label, colspan)``. An empty
    label renders as a spacer cell with no label.
    """
    th_cells = [h if isinstance(h, Tag) else ui.tags.th(h) for h in headers]
    thead_rows: list[Any] = []
    if group_headers:
        group_cells: list[Any] = []
        for label, span in group_headers:
            if label:
                group_cells.append(
                    ui.tags.th(
                        label,
                        colspan=str(span),
                        style=(
                            "text-align:center;background:#eef3f8;"
                            "color:#132747;font-size:0.72rem;"
                            "letter-spacing:0.04em;text-transform:uppercase;"
                            "border-bottom:2px solid #c5d2e0;"
                        ),
                    )
                )
            else:
                group_cells.append(
                    ui.tags.th("", colspan=str(span), style="background:transparent;")
                )
        thead_rows.append(ui.tags.tr(*group_cells))
    thead_rows.append(ui.tags.tr(*th_cells))
    tbody_rows: list[Any] = []
    for row in rows:
        td_cells = [
            ui.tags.td(cell) if isinstance(cell, Tag) else ui.tags.td(str(cell))
            for cell in row
        ]
        tbody_rows.append(ui.tags.tr(*td_cells))
    return ui.tags.table(
        ui.tags.thead(*thead_rows),
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


def _fmt_stat(v: Any, cat: str = "", per_game: bool = False) -> str:
    """Format a stat value for display. Returns '—' for None/NaN.

    Args:
        v: The stat value to format.
        cat: Lowercase category key (e.g. "hr", "whip", "avg").
             When provided, formatting is category-aware.
        per_game: When True, categories that are normally integer counting
            stats (HR/SB/BB/K/W/SV+H) are treated as per-game rates and
            rendered with 2 decimal places. This prevents waiver/roster
            displays from rounding "0.29 HR/game" down to "0".
    """
    if v is None:
        return "—"
    if isinstance(v, float):
        import math

        if math.isnan(v):
            return "—"
        cat_lower = cat.lower()
        if cat_lower in _INTEGER_CATS:
            if per_game:
                return f"{v:.2f}"
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
    "xwoba",
    "woba",
    "barrel_pct",
    "hard_hit_pct",
    "avg_launch_angle",
    "sweet_spot_pct",
    "bat_speed_pctile",
    "xera",
    "xwoba_against",
    "k_bb_pct",
    "barrel_pct_against",
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
                    COALESCE(s.sv_h, 0)  AS sv_h,
                    a.xwoba              AS xwoba,
                    a.woba               AS woba,
                    a.barrel_pct         AS barrel_pct,
                    a.hard_hit_pct       AS hard_hit_pct,
                    a.avg_launch_angle   AS avg_launch_angle,
                    a.sweet_spot_pct     AS sweet_spot_pct,
                    a.bat_speed_pctile   AS bat_speed_pctile,
                    a.xera               AS xera,
                    a.xwoba_against      AS xwoba_against,
                    a.k_bb_pct           AS k_bb_pct,
                    a.barrel_pct_against AS barrel_pct_against
                FROM {FACT_ROSTERS} r
                LEFT JOIN {DIM_PLAYERS} p ON r.player_id = p.player_id
                LEFT JOIN fact_player_advanced_stats a
                    ON r.player_id = a.player_id
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
                    adv_df, _, _ = _load_advanced_with_league_avgs()
                    df = annotate_with_streaks(
                        df, daily_df, advanced_df=adv_df if not adv_df.empty else None
                    )
                elif "streak" not in df.columns:
                    df["streak"] = "—"
                return df
    except (duckdb.Error, Exception) as exc:
        logger.warning("Could not load roster from DB: %s", exc)
    return _empty_roster_df()


def _filter_inactive_waiver_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop waiver rows for players who can't contribute this week.

    A player is considered inactive when either:
      * they have zero games played season-to-date (no stats, no signal), or
      * their position string contains an ``NA`` or ``IL`` eligibility tag.

    Both cases produce rows with all-zero rates and no streak signal,
    which otherwise pad the bottom of the waiver list with noise.
    """
    if df.empty:
        return df

    out = df
    if "games_played" in out.columns:
        gp = pd.to_numeric(out["games_played"], errors="coerce").fillna(0)
        out = out[gp > 0]

    if "position" in out.columns:

        def _is_inactive_position(pos: object) -> bool:
            tokens = {t.strip().upper() for t in str(pos or "").split(",")}
            return bool(tokens & {"NA", "IL", "IL10", "IL15", "IL60"})

        out = out[~out["position"].apply(_is_inactive_position)]

    return out.reset_index(drop=True)


def _waiver_df_from_report(report: dict[str, Any]) -> pd.DataFrame:
    """Build the waiver wire DataFrame directly from the daily report.

    The pipeline computes full FA rankings (with names, positions, and
    per-game stats) and embeds them in ``daily_report["waiver_rankings"]``.
    Reading from the report avoids a second DB query and ensures the UI
    is always consistent with the report's adds/drops section.
    """
    rankings = report.get("waiver_rankings") if isinstance(report, dict) else None
    if not isinstance(rankings, list) or not rankings:
        return pd.DataFrame()

    df = pd.DataFrame(rankings)
    if df.empty:
        return df

    # Sort by overall_score (should already be sorted, but be defensive).
    if "overall_score" in df.columns:
        df = df.sort_values("overall_score", ascending=False).reset_index(drop=True)

    df.insert(0, "rank", range(1, len(df) + 1))
    df["score"] = df.get("overall_score", 0.0)
    df["callup"] = df.get("is_callup", False)
    df["from_level"] = None

    # Annotate streaks if we have recent daily stats.
    try:
        daily_df = _load_recent_daily_stats()
        if not daily_df.empty:
            adv_df, _, _ = _load_advanced_with_league_avgs()
            df = annotate_with_streaks(
                df, daily_df, advanced_df=adv_df if not adv_df.empty else None
            )
        else:
            df["streak"] = "—"
    except Exception as exc:
        logger.warning("Streak annotation failed: %s", exc)
        df["streak"] = "—"

    return df


# Advanced-stat columns by player type — keys must match
# fact_player_advanced_stats column names.
_HITTER_ADV_COLS: tuple[str, ...] = (
    "xwoba",
    "woba",
    "barrel_pct",
    "hard_hit_pct",
    "avg_launch_angle",
    "sweet_spot_pct",
    "bat_speed_pctile",
)
_PITCHER_ADV_COLS: tuple[str, ...] = (
    "xera",
    "xwoba_against",
    "k_bb_pct",
    "barrel_pct_against",
)
# Stats where lower is better — coloring is inverted.
_ADV_LOWER_BETTER: frozenset[str] = frozenset(
    {"xera", "xwoba_against", "barrel_pct_against"}
)


def _load_advanced_with_league_avgs() -> tuple[
    pd.DataFrame, dict[str, float], dict[str, float]
]:
    """Load fact_player_advanced_stats and per-cohort league averages.

    Splits hitters from pitchers by which stat is populated (xwoba marks a
    hitter, xera a pitcher) and returns the means of each cohort. Empty
    DataFrames and empty dicts are returned when nothing is loaded so the
    caller can fall back gracefully.
    """
    # Pull current season stats, but COALESCE each column against the
    # previous season's value so early-season rows (where Savant hasn't
    # cleared its min-sample thresholds for Barrel%/HardHit%/etc.) still
    # get a defensible value to display. Batted-ball metrics are sticky
    # year-over-year so the prior-season fallback is reasonable.
    try:
        with managed_connection() as conn:
            adv: pd.DataFrame = conn.execute(
                f"""
                WITH max_season AS (
                    SELECT MAX(season) AS s FROM {FACT_PLAYER_ADVANCED_STATS}
                ),
                curr AS (
                    SELECT * FROM {FACT_PLAYER_ADVANCED_STATS}
                    WHERE season = (SELECT s FROM max_season)
                ),
                prev AS (
                    SELECT * FROM {FACT_PLAYER_ADVANCED_STATS}
                    WHERE season = (SELECT s - 1 FROM max_season)
                )
                SELECT
                    COALESCE(curr.player_id, prev.player_id)       AS player_id,
                    COALESCE(curr.xwoba, prev.xwoba)               AS xwoba,
                    COALESCE(curr.woba, prev.woba)                 AS woba,
                    COALESCE(curr.barrel_pct, prev.barrel_pct)     AS barrel_pct,
                    COALESCE(curr.hard_hit_pct, prev.hard_hit_pct) AS hard_hit_pct,
                    COALESCE(curr.avg_launch_angle,
                             prev.avg_launch_angle)                AS avg_launch_angle,
                    COALESCE(curr.sweet_spot_pct,
                             prev.sweet_spot_pct)                  AS sweet_spot_pct,
                    COALESCE(curr.bat_speed_pctile,
                             prev.bat_speed_pctile)                AS bat_speed_pctile,
                    COALESCE(curr.xera, prev.xera)                 AS xera,
                    COALESCE(curr.xwoba_against,
                             prev.xwoba_against)                   AS xwoba_against,
                    COALESCE(curr.k_bb_pct, prev.k_bb_pct)         AS k_bb_pct,
                    COALESCE(curr.barrel_pct_against,
                             prev.barrel_pct_against)              AS barrel_pct_against
                FROM curr
                FULL OUTER JOIN prev ON curr.player_id = prev.player_id
                """
            ).fetchdf()
    except (duckdb.Error, Exception) as exc:
        logger.warning("Could not load advanced stats: %s", exc)
        return pd.DataFrame(), {}, {}

    if adv.empty:
        return adv, {}, {}

    hitter_mask = adv["xwoba"].notna()
    pitcher_mask = adv["xera"].notna()
    hitter_means: dict[str, float] = {}
    pitcher_means: dict[str, float] = {}
    if hitter_mask.any():
        hitter_means = {
            c: float(adv.loc[hitter_mask, c].mean(skipna=True))
            for c in _HITTER_ADV_COLS
            if c in adv.columns
        }
    if pitcher_mask.any():
        pitcher_means = {
            c: float(adv.loc[pitcher_mask, c].mean(skipna=True))
            for c in _PITCHER_ADV_COLS
            if c in adv.columns
        }
    return adv, hitter_means, pitcher_means


def _color_adv(
    value: Any,
    avg: float | None,
    *,
    lower_better: bool = False,
    digits: int = 3,
    neutral_pct: float = 0.05,
) -> Any:
    """Render an advanced-stat value coloured by its delta from league average.

    - Green when the player is at least ``neutral_pct`` better than league.
    - Yellow within the neutral band.
    - Red when at least ``neutral_pct`` worse.
    Falls back to '—' for missing values or missing averages.
    """
    if value is None or avg is None or (isinstance(avg, float) and pd.isna(avg)):
        return _fmt_adv(value, digits)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(v):
        return "—"
    if not avg or not pd.notna(avg):
        return _fmt_adv(v, digits)
    delta = (v - float(avg)) / float(avg)
    if lower_better:
        delta = -delta
    if delta > neutral_pct:
        color = "#2e7d32"  # green — better
    elif delta < -neutral_pct:
        color = "#c62828"  # red — worse
    else:
        color = "#b8860b"  # amber — about average
    return ui.tags.span(f"{v:.{digits}f}", style=f"color:{color};font-weight:600;")


_TRANSACTIONS_COLS = [
    "mlb_id",
    "full_name",
    "team",
    "position",
    "txn_type",
    "transaction_date",
    "description",
]


def _load_transactions() -> pd.DataFrame:
    """Fetch recent MLB transactions (call-ups, demotions, IL moves, DFAs, releases).

    Calls the MLB Stats API directly via mlb_client.get_mlb_transactions().
    Enriches call-ups and IL activations with a brief scout report.
    """
    from src.api.mlb_client import get_mlb_transactions

    try:
        df = get_mlb_transactions(days=3)
        if not df.empty:
            df = _enrich_scout_notes(df)
            return df
    except Exception as exc:
        logger.warning("Could not fetch MLB transactions: %s", exc)
    return pd.DataFrame(columns=_TRANSACTIONS_COLS + ["scout_note"])


def _enrich_scout_notes(txn_df: pd.DataFrame) -> pd.DataFrame:
    """Add scout_note column for call-ups and IL activations.

    Cross-references mlb_id with dim_players → fact_player_advanced_stats
    to generate a 2-3 sentence scouting assessment. For call-ups, also
    fetches MiLB stats and player bio (age, draft year, debut).
    """
    from src.api.mlb_client import get_minor_league_stats, get_player_bio_batch

    txn_df = txn_df.copy()
    txn_df["scout_note"] = ""

    # Only scout call-ups and IL activations
    mask = txn_df["txn_type"].isin(["call_up", "il_activation"])
    target_ids = txn_df.loc[mask, "mlb_id"].dropna().astype(int).tolist()
    if not target_ids:
        return txn_df

    # Fetch player bios (age, draft, debut) for call-ups
    callup_mask = txn_df["txn_type"] == "call_up"
    callup_ids = txn_df.loc[callup_mask, "mlb_id"].dropna().astype(int).tolist()
    bio_map: dict[int, dict[str, Any]] = {}
    milb_map: dict[int, dict[str, Any]] = {}
    if callup_ids:
        try:
            bio_map = get_player_bio_batch(callup_ids)
        except Exception as exc:
            logger.debug("Bio batch fetch failed: %s", exc)

        # Fetch MiLB stats for each call-up
        for mid in callup_ids:
            try:
                milb_df = get_minor_league_stats(mid, 2026)
                if not milb_df.empty:
                    # Aggregate across levels — take highest level row
                    top = milb_df.iloc[0]
                    milb_map[mid] = dict(top.to_dict())  # type: ignore[arg-type]
            except Exception:
                pass

    # Look up MLB stats from MotherDuck
    stats_map: dict[int, dict[str, Any]] = {}
    try:
        with managed_connection() as conn:
            placeholders = ",".join(str(int(x)) for x in target_ids)
            rows = conn.execute(f"""
                SELECT
                    p.mlb_id,
                    p.full_name,
                    array_to_string(p.positions, ',') AS positions,
                    s.ab, s.h, s.hr, s.sb, s.avg, s.ops,
                    s.ip, s.w, s.k, s.whip, s.sv_h,
                    a.xwoba, a.barrel_pct, a.hard_hit_pct,
                    a.xera, a.xwoba_against, a.k_bb_pct
                FROM {DIM_PLAYERS} p
                LEFT JOIN (
                    SELECT player_id,
                           SUM(ab) AS ab, SUM(h) AS h, SUM(hr) AS hr,
                           SUM(sb) AS sb,
                           CASE WHEN SUM(ab) > 0
                                THEN ROUND(SUM(h)::DECIMAL / SUM(ab), 3)
                                ELSE NULL END AS avg,
                           CASE WHEN SUM(ab) > 0
                                THEN ROUND(
                                    (SUM(h) + SUM(COALESCE(bb,0))
                                     + SUM(COALESCE(hbp,0)))::DECIMAL
                                    / (SUM(ab) + SUM(COALESCE(bb,0))
                                       + SUM(COALESCE(hbp,0))
                                       + SUM(COALESCE(sf,0)))
                                    + CASE WHEN SUM(ab) > 0
                                           THEN SUM(COALESCE(tb,0))::DECIMAL
                                                / SUM(ab)
                                           ELSE 0 END, 3)
                                ELSE NULL END AS ops,
                           SUM(COALESCE(ip, 0)) AS ip,
                           SUM(COALESCE(w, 0)) AS w,
                           SUM(COALESCE(k, 0)) AS k,
                           CASE WHEN SUM(COALESCE(ip,0)) > 0
                                THEN ROUND(
                                    (SUM(COALESCE(walks_allowed,0))
                                     + SUM(COALESCE(hits_allowed,0)))::DECIMAL
                                    / SUM(ip), 3)
                                ELSE NULL END AS whip,
                           SUM(COALESCE(sv,0))
                           + SUM(COALESCE(holds,0)) AS sv_h
                    FROM {FACT_PLAYER_STATS_DAILY}
                    WHERE stat_date >= '2026-01-01'
                    GROUP BY player_id
                ) s ON s.player_id = p.player_id
                LEFT JOIN {FACT_PLAYER_ADVANCED_STATS} a
                    ON a.player_id = p.player_id AND a.season = 2026
                WHERE p.mlb_id IN ({placeholders})
            """).fetchdf()

            for _, row in rows.iterrows():
                mid = int(row["mlb_id"])
                stats_map[mid] = dict(row.to_dict())  # type: ignore[arg-type]
    except Exception as exc:
        logger.debug("Scout note stats lookup failed: %s", exc)

    # Generate scout notes
    for idx in txn_df.index[mask]:
        mlb_id = int(txn_df.at[idx, "mlb_id"])  # type: ignore[arg-type]
        txn_type = str(txn_df.at[idx, "txn_type"])
        pos = str(txn_df.at[idx, "position"])
        name = str(txn_df.at[idx, "full_name"])

        stats = stats_map.get(mlb_id)
        bio = bio_map.get(mlb_id)
        milb = milb_map.get(mlb_id)
        note = _generate_scout_note(name, pos, txn_type, stats, bio, milb)
        txn_df.at[idx, "scout_note"] = note

    return txn_df


def _generate_scout_note(
    name: str,
    position: str,
    txn_type: str,
    stats: dict[str, Any] | None,
    bio: dict[str, Any] | None = None,
    milb: dict[str, Any] | None = None,
) -> str:
    """Generate a 2-3 sentence scout report for a transaction player."""
    is_pitcher = position in ("P", "SP", "RP", "LHP", "RHP")

    has_mlb_stats = stats is not None and (
        stats.get("ab") not in (None, 0) or stats.get("ip") not in (None, 0)
    )

    if txn_type == "call_up" and not has_mlb_stats:
        # Call-up with no MLB track record — use bio + MiLB stats
        return _callup_scout_note(position, is_pitcher, bio, milb)

    if not has_mlb_stats:
        return "Returning from IL with no 2026 stats on file. Check rehab results before rostering."

    if is_pitcher:
        return _pitcher_scout_note(txn_type, stats)  # type: ignore[arg-type]
    return _hitter_scout_note(txn_type, stats)  # type: ignore[arg-type]


def _callup_scout_note(
    position: str,
    is_pitcher: bool,
    bio: dict[str, Any] | None,
    milb: dict[str, Any] | None,
) -> str:
    """Scout note for a call-up using bio and MiLB stats."""
    parts: list[str] = []

    # Bio context: age, debut status
    age = bio.get("age") if bio else None
    debut = bio.get("debut_date") if bio else None

    import datetime as _dt

    is_debut = debut is None or str(debut) >= str(_dt.date.today())
    if age and is_debut:
        parts.append(f"MLB debut at {age} years old.")
    elif age:
        parts.append(f"Age-{age} return to the bigs.")
    else:
        parts.append("Call-up from minors.")

    # MiLB performance
    if milb:
        level = str(milb.get("level", ""))
        if is_pitcher:
            m_ip = float(milb.get("ip") or 0)
            m_era = milb.get("era")
            m_whip = milb.get("whip")
            m_k = int(milb.get("k") or 0)
            bits: list[str] = []
            if level:
                bits.append(f"{level}:")
            if m_ip > 0:
                k9 = round(m_k / m_ip * 9, 1)
                bits.append(f"{m_ip:.0f} IP, {k9} K/9")
            if m_era is not None:
                bits.append(f"{float(m_era):.2f} ERA")
            if m_whip is not None:
                bits.append(f"{float(m_whip):.2f} WHIP")
            if bits:
                parts.append(f"MiLB: {' '.join(bits)}.")
        else:
            m_avg = milb.get("avg")
            m_ops = milb.get("ops")
            m_hr = int(milb.get("hr") or 0)
            m_sb = int(milb.get("sb") or 0)
            bits = []
            if level:
                bits.append(f"{level}:")
            if m_avg is not None:
                bits.append(f"{m_avg}")
            if m_ops is not None:
                bits.append(f"{m_ops} OPS")
            if m_hr > 0 or m_sb > 0:
                bits.append(f"{m_hr} HR/{m_sb} SB")
            if bits:
                parts.append(f"MiLB: {' '.join(bits)}.")
    elif not parts:
        # No bio, no milb
        if is_pitcher:
            return (
                "Fresh arm from the minors with no MLB track record. "
                "Monitor usage pattern before adding."
            )
        return "Prospect getting first MLB opportunity. Speculative add only in deeper leagues."

    # Verdict
    if milb and is_pitcher:
        m_era = float(milb.get("era") or 99)
        m_whip = float(milb.get("whip") or 99)
        if m_era <= 3.00 and m_whip <= 1.15:
            parts.append("Dominant minor league numbers — high-priority pickup.")
        elif m_era <= 4.00:
            parts.append("Solid MiLB ratios; worth a speculative add.")
        else:
            parts.append("Fringe numbers; stream only if desperate for innings.")
    elif milb and not is_pitcher:
        m_ops_val = float(milb.get("ops") or 0)
        if m_ops_val >= 0.900:
            parts.append("Mashing in the minors — priority add in all formats.")
        elif m_ops_val >= 0.750:
            parts.append("Productive MiLB bat; worth a roster flier.")
        else:
            parts.append("Light bat at upper levels; deep-league only.")

    return " ".join(parts[:3])


def _hitter_scout_note(txn_type: str, stats: dict[str, Any]) -> str:
    """Scout note for a hitter based on available stats."""
    parts: list[str] = []
    avg = stats.get("avg")
    ops = stats.get("ops")
    hr = stats.get("hr", 0) or 0
    sb = stats.get("sb", 0) or 0
    xwoba = stats.get("xwoba")
    barrel = stats.get("barrel_pct")
    hard_hit = stats.get("hard_hit_pct")

    # Opener: context on transaction type
    if txn_type == "call_up":
        parts.append("Call-up worth monitoring.")
    else:
        parts.append("Back from IL and ready to contribute.")

    # Performance summary
    perf_bits: list[str] = []
    if avg is not None and float(avg) > 0:
        perf_bits.append(f".{int(float(avg) * 1000):03d}")
    if ops is not None and float(ops) > 0:
        perf_bits.append(f"{float(ops):.3f} OPS")
    if hr > 0 or sb > 0:
        perf_bits.append(f"{int(hr)} HR/{int(sb)} SB")
    if perf_bits:
        parts.append(f"Season line: {', '.join(perf_bits)}.")

    # Savant quality assessment
    quality_flags: list[str] = []
    if xwoba is not None and float(xwoba) >= 0.340:
        quality_flags.append("elite xwOBA")
    elif xwoba is not None and float(xwoba) >= 0.320:
        quality_flags.append("solid xwOBA")
    if barrel is not None and float(barrel) >= 10:
        quality_flags.append("high barrel rate")
    if hard_hit is not None and float(hard_hit) >= 45:
        quality_flags.append("premium exit velos")

    if quality_flags:
        parts.append(f"Savant flags: {', '.join(quality_flags)} — roster-worthy bat.")
    elif xwoba is not None and float(xwoba) < 0.290:
        parts.append(
            "Below-average quality of contact. Pass unless desperate for position fill."
        )
    else:
        parts.append("Moderate upside; stream if position-needy.")

    return " ".join(parts[:3])


def _pitcher_scout_note(txn_type: str, stats: dict[str, Any]) -> str:
    """Scout note for a pitcher based on available stats."""
    parts: list[str] = []
    ip = float(stats.get("ip") or 0)
    whip = stats.get("whip")
    k = int(stats.get("k") or 0)
    xera = stats.get("xera")
    xwoba_ag = stats.get("xwoba_against")
    k_bb_pct = stats.get("k_bb_pct")
    sv_h = int(stats.get("sv_h") or 0)

    # Opener
    if txn_type == "call_up":
        parts.append("Arm called up from minors.")
    else:
        parts.append("Returning from IL — slot back in if ratios are intact.")

    # Workload + ratios
    perf_bits: list[str] = []
    if ip > 0:
        k_per_9 = round(k / ip * 9, 1) if ip > 0 else 0
        perf_bits.append(f"{ip:.0f} IP, {k_per_9} K/9")
    if whip is not None and float(whip) > 0:
        perf_bits.append(f"{float(whip):.2f} WHIP")
    if sv_h > 0:
        perf_bits.append(f"{sv_h} SV+H")
    if perf_bits:
        parts.append(f"Season: {', '.join(perf_bits)}.")

    # Quality flags
    quality_flags: list[str] = []
    if xera is not None and float(xera) <= 3.20:
        quality_flags.append("elite xERA")
    elif xera is not None and float(xera) <= 3.80:
        quality_flags.append("solid xERA")
    if k_bb_pct is not None and float(k_bb_pct) >= 20:
        quality_flags.append("strong K-BB%")
    if xwoba_ag is not None and float(xwoba_ag) <= 0.290:
        quality_flags.append("suppresses hard contact")

    if quality_flags:
        parts.append(f"Savant: {', '.join(quality_flags)}. Prioritize pickup.")
    elif xera is not None and float(xera) > 4.50:
        parts.append(
            "Underlying metrics are poor. Avoid unless streaming for W/K volume."
        )
    else:
        parts.append("Serviceable arm; consider streaming if ratios are stable.")

    return " ".join(parts[:3])


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

    News for players on my team is flagged via ``is_mine`` and sorted to
    the top so the most relevant headlines surface first.

    Args:
        days: Number of days back to load news for.
    """
    team_key = _get_my_team_key()
    try:
        with managed_connection() as conn:
            df: pd.DataFrame = conn.execute(
                f"""
                WITH my_latest AS (
                    SELECT MAX(snapshot_date) AS snap
                    FROM {FACT_ROSTERS}
                    WHERE team_id = ?
                ),
                my_players AS (
                    SELECT DISTINCT player_id
                    FROM {FACT_ROSTERS}
                    WHERE team_id = ?
                      AND snapshot_date = (SELECT snap FROM my_latest)
                )
                SELECT
                    n.id, n.player_id, n.player_name, n.headline, n.url,
                    n.source, n.published_at, n.sentiment_label, n.sentiment_score,
                    (mp.player_id IS NOT NULL) AS is_mine
                FROM {FACT_PLAYER_NEWS} n
                LEFT JOIN my_players mp ON mp.player_id = n.player_id
                WHERE n.fetched_at >= NOW() - INTERVAL '{days} days'
                ORDER BY is_mine DESC, n.published_at DESC
                LIMIT 200
                """,
                [team_key, team_key],
            ).fetchdf()
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
        df = _waiver_df_from_report(daily_report())
        if df.empty:
            return df
        # Hide inactive players: no games played yet OR flagged NA/IL.
        # These rows drag down the list with all-zero stats and no signal.
        df = _filter_inactive_waiver_rows(df)
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
    def advanced_stats_bundle() -> tuple[
        pd.DataFrame, dict[str, float], dict[str, float]
    ]:
        _refresh_counter()
        return _load_advanced_with_league_avgs()

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
            # Prefer the name stored in the add dict itself (set by the
            # pipeline); fall back to the id→name lookup, then to the raw id.
            add_name = str(a.get("add_name", "")) or id_to_name.get(add_id, add_id)
            drop_name = str(a.get("drop_name", "")) or id_to_name.get(drop_id, drop_id)
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

            # Determine pitcher vs batter for projection display.
            # Prefer the explicit flag from the pipeline; fall back to
            # parsing the position string on both "," and "/" separators.
            add_is_pitcher = bool(
                a.get("add_is_pitcher")
                if "add_is_pitcher" in a
                else _position_is_pitcher(add_pos)
            )
            drop_is_pitcher = bool(
                a.get("drop_is_pitcher")
                if "drop_is_pitcher" in a
                else _position_is_pitcher(drop_pos)
            )

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

            # Category breakdown: per-category weighted z-contribution +
            # matchup status. This is the "why" data that explains each
            # recommended add.
            raw_breakdown = a.get("category_breakdown", [])
            breakdown_rows: list[Any] = []
            if isinstance(raw_breakdown, list):
                status_color = {
                    "flippable_win": "#4fc3f7",
                    "flippable_loss": "#ef5350",
                    "toss_up": "#ffb74d",
                    "safe_win": "#66bb6a",
                    "safe_loss": "#616161",
                }
                status_short = {
                    "flippable_win": "leading",
                    "flippable_loss": "trailing",
                    "toss_up": "toss-up",
                    "safe_win": "safe win",
                    "safe_loss": "safe loss",
                }
                for entry in raw_breakdown:
                    if not isinstance(entry, dict):
                        continue
                    cat_key = str(entry.get("category", ""))
                    weighted_z = float(entry.get("weighted_z", 0.0))
                    status = str(entry.get("status", "toss_up"))
                    label = _CATEGORY_META.get(cat_key, {}).get(
                        "label", cat_key.upper()
                    )
                    sign = "+" if weighted_z >= 0 else ""
                    contrib_color = (
                        "#66bb6a"
                        if weighted_z >= 0.5
                        else "#ef5350"
                        if weighted_z <= -0.5
                        else "#a0b4c8"
                    )
                    breakdown_rows.append(
                        ui.tags.div(
                            ui.tags.span(
                                str(label),
                                style=(
                                    "display:inline-block;min-width:48px;"
                                    "font-size:0.72rem;font-weight:700;color:#e8eef5;"
                                ),
                            ),
                            ui.tags.span(
                                f"{sign}{weighted_z:.2f}",
                                style=(
                                    "display:inline-block;min-width:52px;"
                                    f"font-size:0.72rem;font-weight:700;color:{contrib_color};"
                                ),
                            ),
                            ui.tags.span(
                                status_short.get(status, status),
                                style=(
                                    f"font-size:0.68rem;color:{status_color.get(status, '#888')};"
                                    "text-transform:uppercase;letter-spacing:0.04em;"
                                ),
                            ),
                            style="display:flex;gap:10px;align-items:center;padding:2px 0;",
                        )
                    )

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
                # Collapsible "why" panel: per-category contribution breakdown.
                # Only render the dropdown when there's actual breakdown data.
                ui.tags.span()
                if not breakdown_rows
                else ui.tags.details(
                    ui.tags.summary(
                        "Why this pick? ▾",
                        style=(
                            "cursor:pointer;font-size:0.72rem;color:#7ab8d4;"
                            "font-weight:700;list-style:none;margin-bottom:4px;"
                        ),
                    ),
                    ui.tags.div(
                        ui.tags.div(
                            "Per-category contribution (weighted z-score). "
                            "Positive = this pickup helps the category; "
                            "flippable/toss-up categories are weighted higher "
                            "because they swing the match.",
                            style=(
                                "font-size:0.68rem;color:#8ab;margin-bottom:6px;"
                                "line-height:1.35;"
                            ),
                        ),
                        *breakdown_rows,
                        style=(
                            "background:#09182c;border:1px solid #1e3a5f;"
                            "border-radius:4px;padding:6px 10px;margin-top:2px;"
                        ),
                    ),
                    style="margin-top:8px;",
                ),
                style=(
                    "background:#0d1f38;border:1px solid #1e3a5f;border-radius:6px;"
                    "padding:0.75rem 1rem;overflow:hidden;"
                ),
            )
            cards.append(card)

        return ui.tags.div(
            *cards,
            style=(
                "display:grid;"
                "grid-template-columns:repeat(auto-fill, minmax(380px, 1fr));"
                "gap:0.75rem;"
                "padding:0.25rem 0;"
            ),
        )

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

        adv_df, hit_avgs, pit_avgs = advanced_stats_bundle()
        if not adv_df.empty:
            df = df.merge(
                adv_df,
                on="player_id",
                how="left",
                suffixes=("", "_adv"),
            )

        def _is_p(row: pd.Series) -> bool:
            pos = str(row.get("position", ""))
            return bool(row.get("is_pitcher", False)) or any(
                p in pos.upper() for p in ("SP", "RP")
            )

        type_mask = df.apply(_is_p, axis=1)
        hitters_df = df[~type_mask].copy()
        pitchers_df = df[type_mask].copy()

        def _common_cells(r: pd.Series) -> list[Any]:
            callup_badge = (
                ui.tags.span(
                    " ⬆",
                    style="background:#f5a623;color:#132747;border-radius:3px;"
                    "padding:1px 4px;font-size:0.68rem;font-weight:700;margin-left:4px;",
                )
                if r.get("callup")
                else ui.tags.span()
            )
            score_val = float(r.get("score", 0) or 0)
            fit_val = float(r.get("fit_score", 0) or 0)
            fit_color = (
                "#2e7d32" if fit_val >= 2.0 else "#e65100" if fit_val >= 0.5 else "#888"
            )
            return [
                str(int(r.get("rank", 0))),
                ui.tags.span(str(r.get("player_name", "")), callup_badge),
                str(r.get("team", "")),
                str(r.get("position", "")),
                ui.tags.span(
                    f"{score_val:.1f}",
                    style="font-weight:700;color:#132747;",
                ),
                ui.tags.span(
                    f"{fit_val:+.1f}",
                    style=f"font-weight:700;color:{fit_color};",
                ),
                _streak_badge(str(r.get("streak", "—"))),
            ]

        hitter_header_labels = [
            "#",
            "Player",
            "Team",
            "Pos",
            "Score",
            "Fit",
            "Streak",
            "AVG",
            "OPS",
            "HR",
            "SB",
            "wOBA",
            "xwOBA",
            "Barrel%",
            "HardHit%",
            "LA",
            "SwSp%",
            "BatSp",
        ]
        hitter_headers: list[Any] = [
            _th_tip(label, _ROSTER_TOOLTIPS.get(label, ""))
            for label in hitter_header_labels
        ]
        hitter_rows: list[list[Any]] = []
        for _, r in hitters_df.iterrows():
            row_cells = _common_cells(r)
            row_cells.extend(
                [
                    _fmt_stat(r.get("avg"), "avg"),
                    _fmt_stat(r.get("ops"), "ops"),
                    _fmt_stat(r.get("hr"), "hr", per_game=True),
                    _fmt_stat(r.get("sb"), "sb", per_game=True),
                    _color_adv(r.get("woba"), hit_avgs.get("woba"), digits=3),
                    _color_adv(r.get("xwoba"), hit_avgs.get("xwoba"), digits=3),
                    _color_adv(
                        r.get("barrel_pct"), hit_avgs.get("barrel_pct"), digits=1
                    ),
                    _color_adv(
                        r.get("hard_hit_pct"),
                        hit_avgs.get("hard_hit_pct"),
                        digits=1,
                    ),
                    _color_adv(
                        r.get("avg_launch_angle"),
                        hit_avgs.get("avg_launch_angle"),
                        digits=1,
                    ),
                    _color_adv(
                        r.get("sweet_spot_pct"),
                        hit_avgs.get("sweet_spot_pct"),
                        digits=1,
                    ),
                    _color_adv(
                        r.get("bat_speed_pctile"),
                        hit_avgs.get("bat_speed_pctile"),
                        digits=0,
                    ),
                ]
            )
            hitter_rows.append(row_cells)

        pitcher_header_labels = [
            "#",
            "Player",
            "Team",
            "Pos",
            "Score",
            "Fit",
            "Streak",
            "K",
            "W",
            "WHIP",
            "SV+H",
            "xERA",
            "xwOBA-A",
            "K-BB%",
            "Brl%-A",
        ]
        pitcher_headers: list[Any] = [
            _th_tip(label, _ROSTER_TOOLTIPS.get(label, ""))
            for label in pitcher_header_labels
        ]
        pitcher_rows: list[list[Any]] = []
        for _, r in pitchers_df.iterrows():
            row_cells = _common_cells(r)
            row_cells.extend(
                [
                    _fmt_stat(r.get("k"), "k", per_game=True),
                    _fmt_stat(r.get("w"), "w", per_game=True),
                    _fmt_stat(r.get("whip"), "whip"),
                    _fmt_stat(r.get("sv_h"), "sv_h", per_game=True),
                    _color_adv(
                        r.get("xera"),
                        pit_avgs.get("xera"),
                        digits=2,
                        lower_better=True,
                    ),
                    _color_adv(
                        r.get("xwoba_against"),
                        pit_avgs.get("xwoba_against"),
                        digits=3,
                        lower_better=True,
                    ),
                    _color_adv(
                        r.get("k_bb_pct"),
                        pit_avgs.get("k_bb_pct"),
                        digits=1,
                    ),
                    _color_adv(
                        r.get("barrel_pct_against"),
                        pit_avgs.get("barrel_pct_against"),
                        digits=1,
                        lower_better=True,
                    ),
                ]
            )
            pitcher_rows.append(row_cells)

        section_style = (
            "font-size:0.82rem;font-weight:700;color:#132747;"
            "margin:0.25rem 0 0.4rem 0;letter-spacing:0.02em;"
        )
        empty_msg = ui.p("No matches.", style="color:#888;padding:0.25rem 0 0.75rem 0;")
        return ui.div(
            ui.div("Hitters", style=section_style),
            _html_table(hitter_headers, hitter_rows) if hitter_rows else empty_msg,
            ui.div("Pitchers", style=section_style + "margin-top:1rem;"),
            _html_table(pitcher_headers, pitcher_rows) if pitcher_rows else empty_msg,
        )

    # ── Roster ────────────────────────────────────────────────────────────

    @render.ui
    def hitter_roster_ui() -> Tag:
        df = roster_data()
        if df.empty:
            return ui.p("No roster data.", style="color:#888;")
        is_pitcher_mask = df["position"].apply(lambda p: _position_is_pitcher(str(p)))
        hitters = df[~is_pitcher_mask].copy()
        if hitters.empty:
            return ui.p("No hitters found.", style="color:#888;")
        hitters["_slot_order"] = hitters["slot"].apply(_roster_slot_order)
        hitters = hitters.sort_values(
            by=["_slot_order", "slot", "player_name"], kind="stable"
        )
        header_labels = [
            "Slot",
            "Player",
            "Pos",
            "H",
            "HR",
            "SB",
            "BB",
            "AVG",
            "OPS",
            "wOBA",
            "xwOBA",
            "Barrel%",
            "HardHit%",
            "LA",
            "SwSp%",
            "BatSp",
            "Streak",
        ]
        headers: list[Any] = [
            _th_tip(label, _ROSTER_TOOLTIPS.get(label, "")) for label in header_labels
        ]
        rows_out: list[list[Any]] = []
        for _, r in hitters.iterrows():
            rows_out.append(
                [
                    str(r.get("slot", "")),
                    str(r.get("player_name", "")),
                    str(r.get("position", "")),
                    _fmt_stat_tier(r.get("h"), "h"),
                    _fmt_stat_tier(r.get("hr"), "hr"),
                    _fmt_stat_tier(r.get("sb"), "sb"),
                    _fmt_stat_tier(r.get("bb"), "bb"),
                    _fmt_stat_tier(r.get("avg"), "avg"),
                    _fmt_stat_tier(r.get("ops"), "ops"),
                    _fmt_adv_tier(r.get("woba"), "woba", 3),
                    _fmt_adv_tier(r.get("xwoba"), "xwoba", 3),
                    _fmt_adv_tier(r.get("barrel_pct"), "barrel_pct", 1),
                    _fmt_adv_tier(r.get("hard_hit_pct"), "hard_hit_pct", 1),
                    _fmt_adv_tier(r.get("avg_launch_angle"), "avg_launch_angle", 1),
                    _fmt_adv_tier(r.get("sweet_spot_pct"), "sweet_spot_pct", 1),
                    _fmt_adv_tier(r.get("bat_speed_pctile"), "bat_speed_pctile", 0),
                    _streak_badge(str(r.get("streak", "—"))),
                ]
            )
        # 3 identity cols, 6 weekly (H..OPS), 7 season Savant, 1 streak.
        group_headers = [("", 3), ("Weekly", 6), ("Seasonal", 7), ("", 1)]
        return ui.div(
            _roster_tier_legend(),
            _html_table(headers, rows_out, group_headers=group_headers),
        )

    @render.ui
    def pitcher_roster_ui() -> Tag:
        df = roster_data()
        if df.empty:
            return ui.p("No roster data.", style="color:#888;")
        is_pitcher_mask = df["position"].apply(lambda p: _position_is_pitcher(str(p)))
        pitchers = df[is_pitcher_mask].copy()
        if pitchers.empty:
            return ui.p("No pitchers found.", style="color:#888;")
        pitchers["_slot_order"] = pitchers["slot"].apply(_roster_slot_order)
        pitchers = pitchers.sort_values(
            by=["_slot_order", "slot", "player_name"], kind="stable"
        )
        header_labels = [
            "Slot",
            "Player",
            "Pos",
            "W",
            "K",
            "WHIP",
            "K/BB",
            "SV+H",
            "xERA",
            "xwOBA-A",
            "K-BB%",
            "Brl%-A",
            "Streak",
        ]
        headers: list[Any] = [
            _th_tip(label, _ROSTER_TOOLTIPS.get(label, "")) for label in header_labels
        ]
        rows_out: list[list[Any]] = []
        for _, r in pitchers.iterrows():
            whip_raw = r.get("whip")
            whip_val_for_tier = whip_raw if whip_raw not in (None, "") else None
            rows_out.append(
                [
                    str(r.get("slot", "")),
                    str(r.get("player_name", "")),
                    str(r.get("position", "")),
                    _fmt_stat_tier(r.get("w"), "w"),
                    _fmt_stat_tier(r.get("k"), "k"),
                    _fmt_stat_tier(whip_val_for_tier, "whip"),
                    _fmt_stat_tier(r.get("k_bb"), "k_bb"),
                    _fmt_stat_tier(r.get("sv_h"), "sv_h"),
                    _fmt_adv_tier(r.get("xera"), "xera", 2),
                    _fmt_adv_tier(r.get("xwoba_against"), "xwoba_against", 3),
                    _fmt_adv_tier(r.get("k_bb_pct"), "k_bb_pct", 1),
                    _fmt_adv_tier(r.get("barrel_pct_against"), "barrel_pct_against", 1),
                    _streak_badge(str(r.get("streak", "—"))),
                ]
            )
        # 3 identity cols, 5 weekly (W..SV+H), 4 season Savant, 1 streak.
        group_headers = [("", 3), ("Weekly", 5), ("Seasonal", 4), ("", 1)]
        return ui.div(
            _roster_tier_legend(),
            _html_table(headers, rows_out, group_headers=group_headers),
        )

    # ── Transactions ──────────────────────────────────────────────────────

    @render.ui
    def transactions_ui() -> Tag:
        df = transactions_data()
        if df.empty:
            return ui.p("No transactions found.", style="color:#888;padding:0.5rem;")
        type_filter = str(input.transaction_type_filter())
        if type_filter != "All" and "txn_type" in df.columns:
            df = df[df["txn_type"] == type_filter]
        if df.empty:
            return ui.p("No transactions found.", style="color:#888;padding:0.5rem;")
        headers = ["Date", "Type", "Player", "Pos", "Scout Report"]
        rows_out: list[list[Any]] = []
        for _, r in df.head(50).iterrows():
            t = str(r.get("txn_type", ""))
            label, color = _TRANSACTION_TYPE_LABELS.get(t, (t, "#333"))
            player_name = str(r.get("full_name", r.get("player_name", "—")))
            team = str(r.get("team", ""))
            pos = str(r.get("position", ""))
            scout_note = str(r.get("scout_note", ""))

            # For rows without scout notes, show the description instead
            note_display = scout_note if scout_note else str(r.get("description", ""))

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
                            f" ({team})" if team else "",
                            style="color:#4a6282;font-size:0.75rem;",
                        ),
                    ),
                    ui.tags.span(
                        pos,
                        style="font-weight:600;font-size:0.78rem;",
                    ),
                    ui.tags.span(
                        note_display,
                        style="font-size:0.73rem;color:#2a3f5f;font-style:italic;",
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

        def _render_card(row: pd.Series) -> Tag:
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

            return ui.div(
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

        def _render_column(title: str, column_df: pd.DataFrame) -> Tag:
            header = ui.div(
                title,
                style="padding:0.5rem 0.85rem;background:#132747;color:#fff;"
                "font-weight:700;font-size:0.82rem;letter-spacing:0.02em;",
            )
            if column_df.empty:
                body: Any = ui.div(
                    "No news.",
                    style="padding:0.75rem 0.85rem;color:#8096b0;"
                    "font-size:0.82rem;background:#fff;",
                )
            else:
                body = [_render_card(r) for _, r in column_df.iterrows()]
            return ui.div(
                header,
                *(body if isinstance(body, list) else [body]),
                style="border:1px solid #d8e1eb;border-radius:6px;overflow:hidden;",
            )

        if "is_mine" in df.columns:
            mine_df = df[df["is_mine"].astype(bool)]
            other_df = df[~df["is_mine"].astype(bool)]
        else:
            mine_df = df.iloc[0:0]
            other_df = df

        return ui.layout_columns(
            _render_column("⭐ My Team", mine_df),
            _render_column("League", other_df),
            col_widths=[6, 6],
        )

    # ── Manual refresh ────────────────────────────────────────────────────

    @reactive.effect
    @reactive.event(input.refresh_btn)
    def _refresh() -> None:
        _refresh_counter.set(_refresh_counter() + 1)
