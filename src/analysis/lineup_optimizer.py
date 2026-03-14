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


def recommend_adds(
    waiver_df: pd.DataFrame,
    my_roster_df: pd.DataFrame,
    acquisitions_used: int,
    config: LeagueSettings,
) -> list[dict[str, object]]:
    """Return the top actionable waiver wire adds within the weekly limit.

    Args:
        waiver_df: Output of rank_free_agents(), sorted by overall_score.
        my_roster_df: Current roster.
        acquisitions_used: Number of adds already made this week.
        config: LeagueSettings for max_acquisitions_per_week (5).

    Returns:
        List of dicts (up to max_acquisitions_per_week - acquisitions_used):
          {"add_player_id": str, "drop_player_id": str, "reason": str,
           "score": float, "categories_improved": list[str]}
    """
    max_adds = config.max_acquisitions_per_week - acquisitions_used
    if max_adds <= 0:
        return []

    results: list[dict[str, object]] = []
    used_drop_ids: set[str] = set()

    for _, row in waiver_df.iterrows():
        if len(results) >= max_adds:
            break

        add_id = str(row.get("player_id", ""))
        drop_id = str(row.get("recommended_drop_id", ""))
        score = float(row.get("overall_score", 0.0))

        if not add_id or not drop_id:
            continue

        # Avoid recommending the same drop twice
        if drop_id in used_drop_ids:
            continue

        # Parse categories_improved from category_scores JSON
        categories_improved: list[str] = []
        cat_scores_raw = row.get("category_scores", "{}")
        try:
            cat_scores: dict[str, float] = json.loads(str(cat_scores_raw))
            categories_improved = [c for c, v in cat_scores.items() if v > 0]
        except (json.JSONDecodeError, TypeError):
            pass

        reason = (
            f"Adds value in: {', '.join(categories_improved)}"
            if categories_improved
            else "General improvement"
        )

        results.append(
            {
                "add_player_id": add_id,
                "drop_player_id": drop_id,
                "reason": reason,
                "score": score,
                "categories_improved": categories_improved,
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
    }
