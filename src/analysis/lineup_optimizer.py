"""
lineup_optimizer.py

Pure functions for daily lineup optimization and add/drop recommendations.
No DB or API calls. Accepts DataFrames, returns dicts/lists.
"""

from __future__ import annotations

import datetime
import json
from typing import Any

import pandas as pd

from src.config import LeagueSettings

# Active slot types for pitchers
_PITCHER_SLOTS = {"SP", "RP", "P"}
# Flippable/contested statuses where category help is most valuable
_FOCUS_STATUSES = {"flippable_win", "toss_up", "flippable_loss"}


def _position_is_pitcher(pos: str) -> bool:
    """Return True when the position string contains a pitcher slot."""
    parts = {p.strip().upper() for p in pos.replace("/", ",").split(",")}
    return bool(parts & _PITCHER_SLOTS)


def _get_flippable_categories(matchup_df: pd.DataFrame) -> set[str]:
    """Return the set of categories that are still contested."""
    if matchup_df.empty or "status" not in matchup_df.columns:
        return set()
    mask = matchup_df["status"].isin(_FOCUS_STATUSES)
    return set(matchup_df.loc[mask, "category"].tolist())


def _player_helps_flippable(
    player_row: pd.Series[Any], flippable_cats: set[str]
) -> bool:
    """Return True if the player contributes to any flippable category."""
    for cat in flippable_cats:
        val = player_row.get(cat, 0)
        if pd.notna(val) and float(val) > 0:
            return True
    return False


def _player_eligible_for_slot(positions: list[str], slot: str) -> bool:
    """Return True if a player with the given eligible positions can fill the slot."""
    if slot == "Util":
        batter_positions = {"C", "1B", "2B", "3B", "SS", "OF", "Util"}
        return bool(set(positions) & batter_positions)
    return slot in positions


def optimize_daily_lineup(
    roster_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    matchup_df: pd.DataFrame,
    config: LeagueSettings,
) -> dict[str, str]:
    """Assign the best available player to each active roster slot for today.

    Rules:
    1. Only start players who have a game today (present in schedule_df)
    2. Players in IL or NA slots are ineligible
    3. BN players can be moved to active slots
    4. For Util slots: prefer players who help in flippable categories
       (from matchup_df: status in {"flippable_win","toss_up","flippable_loss"})
    5. Pitching: if accumulated IP > 19, be conservative (only start SPs with
       good matchup; prefer saves/holds over IP if close to 21 minimum)

    Args:
        roster_df: Full roster. Columns: player_id, slot, eligible_positions,
                   games_today (bool), accumulated_ip (float).
        schedule_df: Today's game schedule. Columns: player_id, opponent, home_away.
        matchup_df: Output of score_categories().
        config: LeagueSettings.

    Returns:
        Dict mapping slot → player_id. e.g. {"C": "p123", "1B": "p456", ...}
        All active slots must be filled (use best available or bench player).
    """
    active_slots = config.active_positions
    flippable_cats = _get_flippable_categories(matchup_df)

    # Players with games today
    players_with_games: set[str] = set()
    if not schedule_df.empty and "player_id" in schedule_df.columns:
        players_with_games = set(schedule_df["player_id"].astype(str))

    # Eligible players: not in IL/NA
    if roster_df.empty or "player_id" not in roster_df.columns:
        return {}
    eligible_df = roster_df[~roster_df["slot"].isin({"IL", "NA"})].copy()

    # Parse eligible_positions into lists
    def parse_positions(ep: object) -> list[str]:
        if isinstance(ep, list):
            return [str(p) for p in ep]
        if isinstance(ep, str):
            return [p.strip() for p in ep.split(",")]
        return []

    eligible_df = eligible_df.copy()
    eligible_df["_positions"] = eligible_df["eligible_positions"].apply(parse_positions)

    # Determine accumulated IP for pitching conservatism check
    acc_ip_col = "accumulated_ip"
    ip_conservative = False
    if acc_ip_col in eligible_df.columns:
        total_ip = eligible_df[acc_ip_col].fillna(0).sum()
        ip_conservative = float(total_ip) > 19

    # Build slot keys with occurrence suffixes for duplicate slots.
    # First occurrence of a slot uses the bare name; subsequent use slot_2, slot_3 etc.
    def _make_slot_key(slot_name: str, occurrence: int) -> str:
        return slot_name if occurrence == 1 else f"{slot_name}_{occurrence}"

    # Build ordered list of (slot_key, slot_name) respecting Util-last rule.
    non_util: list[tuple[str, str]] = []
    util_list: list[tuple[str, str]] = []
    slot_counters: dict[str, int] = {}
    for s in active_slots:
        slot_counters[s] = slot_counters.get(s, 0) + 1
        key = _make_slot_key(s, slot_counters[s])
        if s == "Util":
            util_list.append((key, s))
        else:
            non_util.append((key, s))
    ordered_slots_keyed = non_util + util_list

    # Build assignment greedily
    lineup: dict[str, str] = {}
    assigned_players: set[str] = set()

    for slot_key, slot in ordered_slots_keyed:
        # Find candidates eligible for this slot.
        # Build a boolean mask by calling the module-level helper for each player.
        slot_mask = [
            _player_eligible_for_slot(list(row["_positions"]), slot)
            for _, row in eligible_df.iterrows()
        ]
        candidates = eligible_df[slot_mask].copy()
        # Remove already-assigned players
        candidates = candidates[~candidates["player_id"].isin(assigned_players)]

        # Only players with games today
        has_game = candidates["player_id"].astype(str).isin(players_with_games)
        candidates_with_game = candidates[has_game]

        if candidates_with_game.empty:
            # Fallback: use any eligible player even without a game
            pool = candidates
        else:
            pool = candidates_with_game

        if pool.empty:
            continue

        # Scoring to select best player
        best_player_id: str | None = None

        if slot == "Util":
            # Prefer players who help flippable categories
            flippable_mask = [
                _player_helps_flippable(row, flippable_cats)
                for _, row in pool.iterrows()
            ]
            flippable_players = pool[flippable_mask]
            if not flippable_players.empty:
                best_player_id = str(flippable_players.iloc[0]["player_id"])
            elif not pool.empty:
                best_player_id = str(pool.iloc[0]["player_id"])

        elif slot in _PITCHER_SLOTS:
            if ip_conservative:
                # Near IP limit: prefer RP/closers for SV_H over starters adding lots of IP
                if slot in {"RP", "P"}:
                    sv_h_col = "sv_h"
                    if sv_h_col in pool.columns:
                        pool_sorted = pool.sort_values(sv_h_col, ascending=False)
                        best_player_id = str(pool_sorted.iloc[0]["player_id"])
                    else:
                        best_player_id = str(pool.iloc[0]["player_id"])
                else:
                    # SP: only start if good matchup — filter to players with games
                    if not pool.empty:
                        best_player_id = str(pool.iloc[0]["player_id"])
            else:
                best_player_id = str(pool.iloc[0]["player_id"])
        else:
            best_player_id = str(pool.iloc[0]["player_id"])

        if best_player_id is not None:
            lineup[slot_key] = best_player_id
            assigned_players.add(best_player_id)

    return lineup


def _lookup_position(player_id: str, df: pd.DataFrame) -> str:
    """Return the position string for a player from a DataFrame, or '' if not found."""
    if df.empty or "player_id" not in df.columns:
        return ""
    rows = df[df["player_id"] == player_id]
    if rows.empty:
        return ""
    # Prefer "position" column; fall back to "eligible_positions" (list, ndarray, or str).
    import numpy as np

    for col in ("position", "eligible_positions"):
        if col not in rows.columns:
            continue
        pos = rows.iloc[0].get(col, "")
        if isinstance(pos, np.ndarray):
            pos = pos.tolist() if pos.ndim > 0 else pos.item()
        if isinstance(pos, (list, tuple)):
            items = [str(p) for p in pos]
            if items:
                return ",".join(items)
            continue
        if isinstance(pos, str) and pos:
            return pos
    return ""


def _lookup_name(player_id: str, df: pd.DataFrame) -> str:
    """Return the player_name/full_name for a player from a DataFrame, or '' if absent."""
    if df.empty or "player_id" not in df.columns:
        return ""
    rows = df[df["player_id"] == player_id]
    if rows.empty:
        return ""
    for col in ("player_name", "full_name"):
        if col in rows.columns:
            name = rows.iloc[0].get(col, "")
            if name and not (isinstance(name, float) and pd.isna(name)):
                return str(name)
    return ""


def _build_matchup_context(
    categories_improved: list[str],
    matchup_df: pd.DataFrame,
) -> str:
    """Build a short matchup-context sentence for the categories being helped.

    e.g. "K: trailing by 18 (flippable) · SV+H: toss-up · HR: safe win"
    """
    if matchup_df.empty or not categories_improved:
        return ""

    cat_label = {
        "h": "H",
        "hr": "HR",
        "sb": "SB",
        "bb": "BB",
        "fpct": "FPCT",
        "avg": "AVG",
        "ops": "OPS",
        "w": "W",
        "k": "K",
        "whip": "WHIP",
        "k_bb": "K/BB",
        "sv_h": "SV+H",
    }
    status_short = {
        "safe_win": "safe win ✓",
        "flippable_win": "leading (flippable)",
        "toss_up": "toss-up",
        "flippable_loss": "trailing (flippable)",
        "safe_loss": "safe loss ✗",
    }

    status_map: dict[str, tuple[str, float, float]] = {}
    for _, r in matchup_df.iterrows():
        cat = str(r.get("category", ""))
        status = str(r.get("status", ""))
        my_val = float(r.get("my_value", 0.0))
        opp_val = float(r.get("opp_value", 0.0))
        status_map[cat] = (status, my_val, opp_val)

    parts: list[str] = []
    for cat in categories_improved:
        if cat not in status_map:
            continue
        status, my_val, opp_val = status_map[cat]
        label = cat_label.get(cat, cat.upper())
        status_str = status_short.get(status, status)
        gap = abs(my_val - opp_val)
        if cat in ("avg", "ops", "fpct", "whip", "k_bb"):
            gap_str = f"{gap:.3f}"
        else:
            gap_str = str(int(round(gap)))
        parts.append(f"{label}: {status_str} (gap {gap_str})")

    return " · ".join(parts)


def recommend_adds(
    waiver_df: pd.DataFrame,
    my_roster_df: pd.DataFrame,
    acquisitions_used: int,
    config: LeagueSettings,
    matchup_df: pd.DataFrame | None = None,
) -> list[dict[str, object]]:
    """Return the top actionable waiver wire adds within the weekly limit.

    Args:
        waiver_df: Output of rank_free_agents(), sorted by overall_score.
            May optionally contain 'position', 'streak', 'is_callup',
            'days_since_callup', 'from_level', and 'player_name' columns.
        my_roster_df: Current roster. May contain 'position' and 'streak'.
        acquisitions_used: Number of adds already made this week.
        config: LeagueSettings for max_acquisitions_per_week (5).
        matchup_df: Optional matchup state for category-level context in reason.

    Returns:
        List of dicts (up to max_acquisitions_per_week - acquisitions_used):
          {
            "add_player_id": str,
            "add_position": str,        # e.g. "OF", "SP"
            "add_streak": str,          # "🔥 Hot" | "❄️ Cold" | "—"
            "add_callup_note": str,     # e.g. "Called up 5 days ago from AAA"
            "drop_player_id": str,
            "drop_position": str,
            "drop_streak": str,
            "reason": str,              # human-readable matchup-aware reason
            "matchup_context": str,     # per-category status for improved cats
            "score": float,
            "categories_improved": list[str],
          }
    """
    max_adds = config.max_acquisitions_per_week - acquisitions_used
    if max_adds <= 0:
        return []

    _mdf = matchup_df if matchup_df is not None else pd.DataFrame()

    results: list[dict[str, object]] = []
    used_drop_ids: set[str] = set()

    for _, row in waiver_df.iterrows():
        if len(results) >= max_adds:
            break

        add_id = str(row.get("player_id", ""))
        score = float(row.get("overall_score", 0.0))
        fit_score = float(row.get("fit_score", 0.0))

        if not add_id:
            continue

        # Re-compute drop recommendation with exclusions so each add gets
        # a unique drop target. This uses the position-aware logic from
        # waiver_ranker (pitcher↔pitcher, hitter↔hitter enforcement).
        from src.analysis.waiver_ranker import find_recommended_drop

        drop_id = find_recommended_drop(
            row, my_roster_df, config, exclude_ids=used_drop_ids
        )

        if not drop_id:
            continue
        if drop_id in used_drop_ids:
            continue

        # Parse categories_improved from category_scores JSON.
        # Prefer categories that are flippable/toss-up (bigger weighted_z
        # values) — take the top contributors by score.
        categories_improved: list[str] = []
        cat_scores: dict[str, float] = {}
        cat_scores_raw = row.get("category_scores", "{}")
        try:
            cat_scores = json.loads(str(cat_scores_raw))
            positive = [(c, v) for c, v in cat_scores.items() if v > 0]
            positive.sort(key=lambda kv: kv[1], reverse=True)
            categories_improved = [c for c, _ in positive]
        except (json.JSONDecodeError, TypeError):
            pass

        # Status lookup for matchup context per category.
        _status_by_cat: dict[str, str] = {}
        if not _mdf.empty and "category" in _mdf.columns and "status" in _mdf.columns:
            for _, mrow in _mdf.iterrows():
                _status_by_cat[str(mrow["category"])] = str(mrow["status"])

        # Top contributions (positive and negative) for detailed breakdown.
        # Rank by absolute z-contribution so the biggest drivers surface
        # whether they help or hurt the pickup.
        breakdown: list[dict[str, object]] = []
        if cat_scores:
            sorted_cats = sorted(
                cat_scores.items(), key=lambda kv: abs(kv[1]), reverse=True
            )
            for cat, weighted_z in sorted_cats:
                if weighted_z == 0:
                    continue
                breakdown.append(
                    {
                        "category": cat,
                        "weighted_z": float(weighted_z),
                        "status": _status_by_cat.get(cat, "toss_up"),
                    }
                )

        # Names from waiver row (preferred) and drop lookup in roster.
        add_name = str(row.get("player_name", "")) or add_id
        drop_name = _lookup_name(drop_id, my_roster_df) or drop_id

        # Position lookup
        add_pos = str(row.get("position", "")) if "position" in row.index else ""
        drop_pos = _lookup_position(drop_id, my_roster_df)

        # Streak
        add_streak = str(row.get("streak", "—")) if "streak" in row.index else "—"
        drop_streak_rows = (
            my_roster_df[my_roster_df["player_id"] == drop_id]
            if not my_roster_df.empty and "player_id" in my_roster_df.columns
            else pd.DataFrame()
        )
        drop_streak = "—"
        if not drop_streak_rows.empty and "streak" in drop_streak_rows.columns:
            drop_streak = str(drop_streak_rows.iloc[0].get("streak", "—"))

        # Call-up note
        callup_note = ""
        is_callup = bool(row.get("is_callup", False))
        if is_callup:
            days = row.get("days_since_callup")
            from_level = str(row.get("from_level", "minors"))
            if days is not None and not (isinstance(days, float) and pd.isna(days)):
                callup_note = f"Called up {int(days)} day(s) ago from {from_level}"
            else:
                callup_note = f"Recent call-up from {from_level}"

        # Matchup context string
        matchup_context = _build_matchup_context(categories_improved, _mdf)

        # Build reason
        cat_labels = {
            "h": "H",
            "hr": "HR",
            "sb": "SB",
            "bb": "BB",
            "fpct": "FPCT",
            "avg": "AVG",
            "ops": "OPS",
            "w": "W",
            "k": "K",
            "whip": "WHIP",
            "k_bb": "K/BB",
            "sv_h": "SV+H",
        }
        cat_str = ", ".join(cat_labels.get(c, c.upper()) for c in categories_improved)
        reason_parts: list[str] = []
        if cat_str:
            reason_parts.append(f"Improves {cat_str}")
        if matchup_context:
            reason_parts.append(matchup_context)
        if add_streak == "🔥 Hot":
            reason_parts.append("Currently on a hot streak")
        if drop_streak == "❄️ Cold":
            reason_parts.append("Drop candidate is in a cold streak")
        if callup_note:
            reason_parts.append(callup_note)

        reason = ". ".join(reason_parts) if reason_parts else "General improvement"

        # Pitcher classification: prefer the waiver row's explicit
        # is_pitcher flag (set by rank_free_agents); fall back to a
        # position-string check that handles both "," and "/" separators.
        add_is_pitcher = bool(
            row.get("is_pitcher", False)
            if "is_pitcher" in row.index
            else _position_is_pitcher(add_pos)
        )
        drop_is_pitcher = _position_is_pitcher(drop_pos)

        results.append(
            {
                "add_player_id": add_id,
                "add_name": add_name,
                "add_position": add_pos,
                "add_is_pitcher": add_is_pitcher,
                "add_streak": add_streak,
                "add_callup_note": callup_note,
                "drop_player_id": drop_id,
                "drop_name": drop_name,
                "drop_position": drop_pos,
                "drop_is_pitcher": drop_is_pitcher,
                "drop_streak": drop_streak,
                "reason": reason,
                "matchup_context": matchup_context,
                "score": score,
                "fit_score": fit_score,
                "categories_improved": categories_improved,
                "category_breakdown": breakdown,
            }
        )
        used_drop_ids.add(drop_id)

    return results


def build_daily_report(
    lineup: dict[str, str],
    adds: list[dict[str, object]],
    matchup_df: pd.DataFrame,
    ip_pace: dict[str, object],
    callup_alerts: list[dict[str, object]],
    report_date: datetime.date | None = None,
    week_number: int = 1,
    waiver_rankings: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build the JSON-serializable daily report consumed by the Shiny app.

    Args:
        lineup: Output of optimize_daily_lineup().
        adds: Output of recommend_adds().
        matchup_df: Output of score_categories().
        ip_pace: Output of check_ip_pace().
        callup_alerts: List of call-up alert dicts.
        report_date: Date of report. Defaults to today.
        week_number: Current week number.

    Returns:
        JSON-serializable dict:
        {
          "report_date": "YYYY-MM-DD",
          "week_number": int,
          "lineup": {"C": "player_id", ...},
          "adds": [{"add_player_id": str, "drop_player_id": str, "reason": str,
                    "score": float, "categories_improved": list[str]}],
          "matchup_summary": [{"category": str, "my_value": float, "opp_value": float,
                                "my_leads": bool, "margin_pct": float, "win_prob": float,
                                "status": str}],
          "ip_pace": {"current_ip": float, "projected_ip": float,
                      "min_ip": int, "on_pace": bool},
          "callup_alerts": [{"player_id": str, "player_name": str,
                              "days_since_callup": int, "team": str, "from_level": str}]
        }
    """
    if report_date is None:
        report_date = datetime.date.today()

    # Convert matchup_df rows to list of dicts for JSON serialization
    matchup_summary: list[dict[str, object]] = []
    if not matchup_df.empty:
        for _, row in matchup_df.iterrows():
            matchup_summary.append(
                {
                    "category": str(row.get("category", "")),
                    "my_value": float(row.get("my_value", 0.0)),
                    "opp_value": float(row.get("opp_value", 0.0)),
                    "my_leads": bool(row.get("my_leads", False)),
                    "margin_pct": float(row.get("margin_pct", 0.0)),
                    "win_prob": float(row.get("win_prob", 0.5)),
                    "status": str(row.get("status", "toss_up")),
                }
            )

    # Extract ip_pace values safely — ip_pace values are object-typed but are floats/ints at runtime
    current_ip = ip_pace.get("current_ip", 0.0)
    projected_ip = ip_pace.get("projected_ip", 0.0)
    min_ip = ip_pace.get("min_ip", 21)
    on_pace = ip_pace.get("on_pace", False)

    return {
        "report_date": report_date.isoformat(),
        "week_number": week_number,
        "lineup": lineup,
        "adds": adds,
        "matchup_summary": matchup_summary,
        "ip_pace": {
            "current_ip": current_ip,
            "projected_ip": projected_ip,
            "min_ip": min_ip,
            "on_pace": on_pace,
        },
        "callup_alerts": callup_alerts,
        "waiver_rankings": waiver_rankings or [],
    }
