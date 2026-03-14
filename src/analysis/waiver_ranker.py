"""
waiver_ranker.py

Pure functions for ranking waiver wire pickups by expected matchup impact.
All functions accept DataFrames, return DataFrames or dicts — no DB or API calls.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.config import LeagueSettings

# Status weight mapping for category scoring
_STATUS_WEIGHTS: dict[str, float] = {
    "safe_win": 0.1,
    "safe_loss": 0.1,
    "toss_up": 1.0,
    "flippable_win": 2.0,
    "flippable_loss": 2.0,
}

# Map from category name to stat column in player rows
_CAT_TO_STAT: dict[str, str] = {
    "h": "h",
    "hr": "hr",
    "sb": "sb",
    "bb": "bb",
    "fpct": "fpct",
    "avg": "avg",
    "ops": "ops",
    "w": "w",
    "k": "k",
    "whip": "whip",
    "k_bb": "k_bb",
    "sv_h": "sv_h",
}


def _get_stat_value(row: pd.Series[Any], cat: str) -> float:
    """Safely get a stat value from a row, defaulting to 0.0."""
    col = _CAT_TO_STAT.get(cat, cat)
    if col in row.index:
        val = row[col]
        if pd.isna(val):
            return 0.0
        return float(val)
    return 0.0


def score_free_agent(
    player_row: pd.Series[Any],
    my_roster_df: pd.DataFrame,
    matchup_df: pd.DataFrame,
    config: LeagueSettings,
) -> dict[str, object]:
    """Score a single free agent by expected matchup impact.

    Args:
        player_row: One row from the free_agents DataFrame. Must have player_id,
                    position columns, and all category stat columns.
        my_roster_df: Current roster. Must have player_id, position, and all
                      category stat columns.
        matchup_df: Output of score_categories(). Used to weight improvements
                    in flippable categories higher.
        config: LeagueSettings instance for win_direction and roster positions.

    Returns:
        {
          "player_id": str,
          "overall_score": float,
          "category_scores": dict[str, float],   # JSON-serializable
          "recommended_drop_id": str,
        }

    Scoring logic:
        - For each category, estimate the delta if we add this player (drop weakest)
        - Weight that delta by status: safe_win/safe_loss=0.1, toss_up=1.0,
          flippable_win/flippable_loss=2.0
        - For WHIP: lower delta is better (invert sign)
        - Sum weighted deltas → overall_score
    """
    player_id = str(player_row.get("player_id", ""))

    # Find who we'd drop
    recommended_drop_id = find_recommended_drop(player_row, my_roster_df, config)

    # Build a status lookup from matchup_df
    status_lookup: dict[str, str] = {}
    if (
        not matchup_df.empty
        and "category" in matchup_df.columns
        and "status" in matchup_df.columns
    ):
        for _, row in matchup_df.iterrows():
            status_lookup[str(row["category"])] = str(row["status"])

    category_scores: dict[str, float] = {}
    overall_score = 0.0

    # For each scoring category, compute the delta from adding this player
    for cat in config.scoring_categories:
        direction = config.category_win_direction.get(cat, "highest")
        weight = _STATUS_WEIGHTS.get(status_lookup.get(cat, "toss_up"), 1.0)

        player_val = _get_stat_value(player_row, cat)

        # Compute what the dropped player contributed
        drop_val = 0.0
        if recommended_drop_id and not my_roster_df.empty:
            drop_rows = my_roster_df[my_roster_df["player_id"] == recommended_drop_id]
            if not drop_rows.empty:
                drop_val = _get_stat_value(drop_rows.iloc[0], cat)

        delta = player_val - drop_val

        # For WHIP: lower is better, so a negative delta (lower WHIP) is good
        if direction == "lowest":
            delta = -delta

        weighted = delta * weight
        category_scores[cat] = weighted
        overall_score += weighted

    return {
        "player_id": player_id,
        "overall_score": overall_score,
        "category_scores": category_scores,
        "recommended_drop_id": recommended_drop_id,
    }


def find_recommended_drop(
    candidate_player: pd.Series[Any],
    my_roster_df: pd.DataFrame,
    config: LeagueSettings,
) -> str:
    """Find the best player to drop when adding the candidate.

    Args:
        candidate_player: The player being considered for add.
                          Must have position eligibility info.
        my_roster_df: Current roster with player_id, eligible_positions,
                      overall_score, and stat columns.
        config: LeagueSettings for position rules.

    Returns:
        player_id of the recommended drop. Must be positionally replaceable
        (same position eligibility as candidate or BN-only player).
        Never drops a player with games_remaining > 0 this week if there
        is a BN-only alternative.
    """
    if my_roster_df.empty:
        return ""

    df = my_roster_df.copy()

    # Get candidate positions
    candidate_positions: list[str] = []
    if "eligible_positions" in candidate_player.index:
        ep = candidate_player["eligible_positions"]
        if isinstance(ep, list):
            candidate_positions = ep
        elif isinstance(ep, str):
            candidate_positions = [p.strip() for p in ep.split(",")]

    # Identify BN-only players (slot is BN and no active slot eligibility)
    bench_slots = set(config.bench_slots)

    def is_bench_only(row: pd.Series[Any]) -> bool:
        slot = str(row.get("slot", row.get("roster_slot", "")))
        return slot in bench_slots

    # Separate bench-only from active players
    bench_candidates = df[df.apply(is_bench_only, axis=1)]
    active_candidates = df[~df.apply(is_bench_only, axis=1)]

    # Prefer dropping BN-only players who have no games remaining
    if not bench_candidates.empty:
        if "games_remaining" in bench_candidates.columns:
            no_games = bench_candidates[
                bench_candidates["games_remaining"].fillna(0) <= 0
            ]
            if not no_games.empty:
                # Drop the bench player with the lowest overall_score
                if "overall_score" in no_games.columns:
                    return str(
                        no_games.loc[no_games["overall_score"].idxmin(), "player_id"]
                    )
                return str(no_games.iloc[0]["player_id"])

        # All bench players are candidates; pick lowest score
        if "overall_score" in bench_candidates.columns:
            return str(
                bench_candidates.loc[
                    bench_candidates["overall_score"].idxmin(), "player_id"
                ]
            )
        return str(bench_candidates.iloc[0]["player_id"])

    # No bench-only players — consider active players at the same position
    if candidate_positions:

        def shares_position(row: pd.Series[Any]) -> bool:
            ep = row.get("eligible_positions", [])
            if isinstance(ep, str):
                ep = [p.strip() for p in ep.split(",")]
            return bool(set(candidate_positions) & set(ep))

        pos_matches = active_candidates[
            active_candidates.apply(shares_position, axis=1)
        ]
        if not pos_matches.empty:
            if "overall_score" in pos_matches.columns:
                return str(
                    pos_matches.loc[pos_matches["overall_score"].idxmin(), "player_id"]
                )
            return str(pos_matches.iloc[0]["player_id"])

    # Fallback: drop lowest-scoring player overall
    if not df.empty:
        if "overall_score" in df.columns:
            return str(df.loc[df["overall_score"].idxmin(), "player_id"])
        return str(df.iloc[0]["player_id"])

    return ""


def rank_free_agents(
    free_agents_df: pd.DataFrame,
    my_roster_df: pd.DataFrame,
    matchup_df: pd.DataFrame,
    callups_df: pd.DataFrame,
    config: LeagueSettings,
) -> pd.DataFrame:
    """Rank all available free agents by expected matchup impact.

    Args:
        free_agents_df: Available players with stats and projections.
        my_roster_df: Current roster.
        matchup_df: Output of score_categories().
        callups_df: Recent call-ups. Columns: player_id, days_since_callup.
        config: LeagueSettings.

    Returns:
        DataFrame sorted descending by overall_score with columns:
          player_id, overall_score, category_scores (JSON string),
          recommended_drop_id, is_callup (bool), days_since_callup (int, NaN if not callup)
    """
    records = []

    callup_ids: set[str] = set()
    callup_days: dict[str, int] = {}
    if not callups_df.empty and "player_id" in callups_df.columns:
        callup_ids = set(callups_df["player_id"].astype(str))
        if "days_since_callup" in callups_df.columns:
            for _, row in callups_df.iterrows():
                callup_days[str(row["player_id"])] = int(row["days_since_callup"])

    for _, player_row in free_agents_df.iterrows():
        scored = score_free_agent(player_row, my_roster_df, matchup_df, config)
        pid = str(scored["player_id"])
        is_callup = pid in callup_ids
        days_since: int | float = callup_days.get(pid, float("nan"))

        records.append(
            {
                "player_id": pid,
                "overall_score": scored["overall_score"],
                "category_scores": json.dumps(scored["category_scores"]),
                "recommended_drop_id": scored["recommended_drop_id"],
                "is_callup": is_callup,
                "days_since_callup": days_since,
            }
        )

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values("overall_score", ascending=False).reset_index(
            drop=True
        )
    return result
