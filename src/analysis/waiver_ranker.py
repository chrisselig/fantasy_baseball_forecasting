"""
waiver_ranker.py

Pure functions for ranking waiver wire pickups by expected matchup impact.
All functions accept DataFrames, return DataFrames or dicts — no DB or API calls.

Scoring model
=============
We use a **z-score value-above-replacement** approach, weighted by matchup
category status. For each scoring category:

    delta      = fa_per_game - drop_per_game    (inverted for "lowest" cats)
    z          = delta / sigma_cat               (sigma_cat from my roster)
    weighted_z = z * status_weight(cat)
    overall    += weighted_z

This normalizes across categories with very different magnitudes (e.g. HR ~1
per game vs K ~7 per game), so no single category dominates. Flippable and
toss-up categories receive higher weights so the ranker reflects *what the
team actually needs to win this week*.

A secondary **fit_score** sums only the contributions from flippable/toss-up
categories — this is the "matchup-aware" score that should drive UI sorting
when the user cares about closing category gaps.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from src.config import LeagueSettings

# Status weight mapping for category scoring.
# Flippable and toss-up categories are weighted far higher than safe ones
# because those are where roster moves actually change outcomes.
_STATUS_WEIGHTS: dict[str, float] = {
    "safe_win": 0.15,
    "safe_loss": 0.15,
    "toss_up": 1.5,
    "flippable_win": 2.5,
    "flippable_loss": 2.5,
}

# Talent vs Fit blend (α). The final overall_score is
#   overall_score = α × talent_score + (1 − α) × fit_score
# where talent_score is the sum of *unweighted* z-scores (matchup-blind value
# above replacement) and fit_score is the matchup-aware subset (flippable +
# toss-up cats). α should be high early in the season (matchup category
# statuses come from tiny samples and are unreliable) and decay as the season
# progresses (late-season standings/needs become decisive). The mapping is
# linear: α = clamp(α_max − (α_max − α_min) × season_progress, α_min, α_max).
_ALPHA_EARLY = 0.55
_ALPHA_LATE = 0.10
_DEFAULT_SEASON_PROGRESS = 0.5  # mid-season fallback

# Categories where "lower is better" (rate stats where we want to minimize)
_LOWER_BETTER: set[str] = {"whip"}

# Pitcher position markers (used for is_pitcher classification)
_PITCHER_POSITIONS: set[str] = {"SP", "RP", "P"}

# Categories and the per-game columns they map to in player rows.
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

# Display-ready key stat columns the UI expects to read off a ranked FA row.
_DISPLAY_STAT_COLS: list[str] = [
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
    "games_played",
]


def _get_stat_value(row: pd.Series[Any], cat: str) -> float:
    """Safely get a stat value from a row, defaulting to 0.0."""
    col = _CAT_TO_STAT.get(cat, cat)
    if col in row.index:
        val = row[col]
        if pd.isna(val):
            return 0.0
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _is_pitcher(positions: Any) -> bool:
    """Return True when the player's positions include SP/RP/P."""
    if positions is None:
        return False
    if isinstance(positions, (list, tuple, np.ndarray)):
        return bool({str(p).upper() for p in positions} & _PITCHER_POSITIONS)
    if isinstance(positions, str):
        parts = [p.strip().upper() for p in positions.replace("/", ",").split(",")]
        return bool(set(parts) & _PITCHER_POSITIONS)
    return False


def _positions_str(positions: Any) -> str:
    """Render positions as a short comma-separated string."""
    if positions is None:
        return ""
    if isinstance(positions, (list, tuple, np.ndarray)):
        return ",".join(str(p) for p in positions)
    if isinstance(positions, str):
        return positions
    return ""


def _compute_category_sigmas(
    roster_df: pd.DataFrame, categories: list[str]
) -> dict[str, float]:
    """Compute a per-category standard deviation from the active roster.

    Uses the roster as a "replacement-level" reference population. A small
    epsilon floor prevents division-by-zero for categories with no variance.
    """
    sigmas: dict[str, float] = {}
    if roster_df.empty:
        return dict.fromkeys(categories, 1.0)

    for cat in categories:
        col = _CAT_TO_STAT.get(cat, cat)
        if col not in roster_df.columns:
            sigmas[cat] = 1.0
            continue
        series = pd.to_numeric(roster_df[col], errors="coerce").dropna()
        if series.empty:
            sigmas[cat] = 1.0
            continue
        std = float(series.std(ddof=0))
        if not math.isfinite(std) or std < 1e-6:
            # No variance — fall back to mean (or 1.0) to keep the scale sensible.
            mean_val = float(series.mean())
            sigmas[cat] = max(abs(mean_val), 1.0)
        else:
            sigmas[cat] = std
    return sigmas


def score_free_agent(
    player_row: pd.Series[Any],
    my_roster_df: pd.DataFrame,
    matchup_df: pd.DataFrame,
    config: LeagueSettings,
    sigmas: dict[str, float] | None = None,
) -> dict[str, object]:
    """Score a single free agent by expected matchup impact.

    Returns dict with:
        player_id, talent_score, fit_score, weighted_score, overall_score,
        category_scores, recommended_drop_id

    ``talent_score`` — sum of *unweighted* z-scores across applicable
    categories. Matchup-blind. Answers "how much above replacement is this
    player overall, regardless of what I need this week?"

    ``fit_score`` — sum of weighted z-scores restricted to flippable /
    toss-up categories. Matchup-aware. Answers "how much does this player
    help me win the categories that are actually contested?"

    ``weighted_score`` — sum of weighted z-scores across all categories
    (the legacy single-number score).

    ``overall_score`` — final blended score, set by ``rank_free_agents``
    after computing α from season progress: ``α × talent + (1−α) × fit``.
    For single-player calls (no cohort context), defaults to ``weighted_score``
    so legacy callers see no change.
    """
    player_id = str(player_row.get("player_id", ""))

    recommended_drop_id = find_recommended_drop(player_row, my_roster_df, config)

    status_lookup: dict[str, str] = {}
    if (
        not matchup_df.empty
        and "category" in matchup_df.columns
        and "status" in matchup_df.columns
    ):
        for _, row in matchup_df.iterrows():
            status_lookup[str(row["category"])] = str(row["status"])

    if sigmas is None:
        sigmas = _compute_category_sigmas(my_roster_df, list(config.scoring_categories))

    category_scores: dict[str, float] = {}
    weighted_score = 0.0
    talent_score = 0.0
    fit_score = 0.0

    # Pitcher/hitter awareness: don't penalize hitters for having 0 WHIP or
    # pitchers for having 0 HR — only consider the categories relevant to
    # the player's role.
    player_is_pitcher = _is_pitcher(
        player_row.get("eligible_positions", player_row.get("positions", []))
    )
    hitter_cats = {"h", "hr", "sb", "bb", "avg", "ops", "fpct"}
    pitcher_cats = {"w", "k", "whip", "k_bb", "sv_h"}

    for cat in config.scoring_categories:
        # Skip categories that don't apply to this player's role
        if player_is_pitcher and cat in hitter_cats:
            category_scores[cat] = 0.0
            continue
        if (not player_is_pitcher) and cat in pitcher_cats:
            category_scores[cat] = 0.0
            continue

        direction = config.category_win_direction.get(cat, "highest")
        status = status_lookup.get(cat, "toss_up")
        weight = _STATUS_WEIGHTS.get(status, 1.0)

        player_val = _get_stat_value(player_row, cat)

        drop_val = 0.0
        if recommended_drop_id and not my_roster_df.empty:
            drop_rows = my_roster_df[my_roster_df["player_id"] == recommended_drop_id]
            if not drop_rows.empty:
                drop_val = _get_stat_value(drop_rows.iloc[0], cat)

        delta = player_val - drop_val
        if direction == "lowest" or cat in _LOWER_BETTER:
            delta = -delta

        sigma = sigmas.get(cat, 1.0) or 1.0
        z = delta / sigma
        weighted = z * weight
        category_scores[cat] = weighted
        weighted_score += weighted
        talent_score += z

        if status in ("flippable_win", "flippable_loss", "toss_up"):
            fit_score += weighted

    return {
        "player_id": player_id,
        # Legacy callers expect overall_score to behave like the old weighted
        # sum; rank_free_agents will overwrite this with the α-blended score.
        "overall_score": weighted_score,
        "weighted_score": weighted_score,
        "talent_score": talent_score,
        "fit_score": fit_score,
        "category_scores": category_scores,
        "recommended_drop_id": recommended_drop_id,
    }


def find_recommended_drop(
    candidate_player: pd.Series[Any],
    my_roster_df: pd.DataFrame,
    config: LeagueSettings,
) -> str:
    """Find the best player to drop when adding the candidate.

    Priority:
      1. Bench players with no games remaining (lowest overall_score first).
      2. Bench players (lowest overall_score first).
      3. Active players sharing the candidate's position eligibility.
      4. Lowest-scoring player overall (last-resort fallback).
    """
    if my_roster_df.empty:
        return ""

    df = my_roster_df.copy()

    candidate_positions: list[str] = []
    ep: Any = None
    if "eligible_positions" in candidate_player.index:
        ep = candidate_player["eligible_positions"]
    elif "positions" in candidate_player.index:
        ep = candidate_player["positions"]
    if isinstance(ep, (list, tuple, np.ndarray)):
        candidate_positions = [str(p) for p in ep]
    elif isinstance(ep, str):
        candidate_positions = [p.strip() for p in ep.split(",")]

    bench_slots = set(config.bench_slots)

    def is_bench_only(row: pd.Series[Any]) -> bool:
        slot = str(row.get("slot", row.get("roster_slot", "")))
        return slot in bench_slots

    bench_candidates = df[df.apply(is_bench_only, axis=1)]
    active_candidates = df[~df.apply(is_bench_only, axis=1)]

    if not bench_candidates.empty:
        if "games_remaining" in bench_candidates.columns:
            no_games = bench_candidates[
                bench_candidates["games_remaining"].fillna(0) <= 0
            ]
            if not no_games.empty:
                if "overall_score" in no_games.columns:
                    return str(
                        no_games.loc[no_games["overall_score"].idxmin(), "player_id"]
                    )
                return str(no_games.iloc[0]["player_id"])

        if "overall_score" in bench_candidates.columns:
            return str(
                bench_candidates.loc[
                    bench_candidates["overall_score"].idxmin(), "player_id"
                ]
            )
        return str(bench_candidates.iloc[0]["player_id"])

    if candidate_positions:

        def shares_position(row: pd.Series[Any]) -> bool:
            ep = row.get("eligible_positions", [])
            if isinstance(ep, (list, tuple, np.ndarray)):
                ep_list = [str(p) for p in ep]
            elif isinstance(ep, str):
                ep_list = [p.strip() for p in ep.split(",")]
            else:
                return False
            return bool(set(candidate_positions) & set(ep_list))

        pos_matches = active_candidates[
            active_candidates.apply(shares_position, axis=1)
        ]
        if not pos_matches.empty:
            if "overall_score" in pos_matches.columns:
                return str(
                    pos_matches.loc[pos_matches["overall_score"].idxmin(), "player_id"]
                )
            return str(pos_matches.iloc[0]["player_id"])

    if not df.empty:
        if "overall_score" in df.columns:
            return str(df.loc[df["overall_score"].idxmin(), "player_id"])
        return str(df.iloc[0]["player_id"])

    return ""


def alpha_from_season_progress(progress: float) -> float:
    """Map season progress (0.0 - 1.0) to the talent/fit blend weight α.

    α decays linearly from ``_ALPHA_EARLY`` (start of season) to
    ``_ALPHA_LATE`` (end of season). Out-of-range inputs are clamped.
    """
    progress = max(0.0, min(1.0, float(progress)))
    return _ALPHA_EARLY - (_ALPHA_EARLY - _ALPHA_LATE) * progress


def rank_free_agents(
    free_agents_df: pd.DataFrame,
    my_roster_df: pd.DataFrame,
    matchup_df: pd.DataFrame,
    callups_df: pd.DataFrame,
    config: LeagueSettings,
    season_progress: float = _DEFAULT_SEASON_PROGRESS,
) -> pd.DataFrame:
    """Rank all available free agents by expected matchup impact.

    The final ``overall_score`` is the α-blend of each FA's matchup-blind
    *talent_score* and matchup-aware *fit_score*. α decays through the
    season (early: trust talent more because category statuses are noisy;
    late: trust fit more because every move matters for standings).

    Args:
        free_agents_df: All available free agents (Yahoo FA query output).
        my_roster_df: User's current roster with per-game rates.
        matchup_df: Output of ``score_categories`` (one row per cat).
        callups_df: Recent MLB call-ups (for the call-up flag/days).
        config: League settings.
        season_progress: 0.0 = opening day, 1.0 = end of regular season.
            Drives α via ``alpha_from_season_progress``. Defaults to 0.5
            (mid-season) when callers don't supply it.

    Returns:
        DataFrame sorted descending by ``overall_score`` with columns:
        player_id, player_name, team, position, is_pitcher,
        overall_score, talent_score, fit_score, weighted_score,
        category_scores (JSON), recommended_drop_id, is_callup,
        days_since_callup, plus per-game stat columns (h, hr, sb, bb, avg,
        ops, w, k, whip, k_bb, sv_h).
    """
    if free_agents_df.empty:
        return pd.DataFrame()

    # Precompute category sigmas once (stable across all FAs).
    sigmas = _compute_category_sigmas(my_roster_df, list(config.scoring_categories))

    callup_ids: set[str] = set()
    callup_days: dict[str, int] = {}
    if not callups_df.empty and "player_id" in callups_df.columns:
        callup_ids = set(callups_df["player_id"].astype(str))
        if "days_since_callup" in callups_df.columns:
            for _, row in callups_df.iterrows():
                callup_days[str(row["player_id"])] = int(row["days_since_callup"])

    records: list[dict[str, Any]] = []
    for _, player_row in free_agents_df.iterrows():
        scored = score_free_agent(
            player_row, my_roster_df, matchup_df, config, sigmas=sigmas
        )
        pid = str(scored["player_id"])
        is_callup = pid in callup_ids
        days_since: int | float = callup_days.get(pid, float("nan"))

        # `positions` is the Yahoo free_agents column; `eligible_positions`
        # is the roster column — accept either.
        raw_positions = player_row.get(
            "eligible_positions", player_row.get("positions", [])
        )
        rec: dict[str, Any] = {
            "player_id": pid,
            "player_name": str(
                player_row.get("full_name", player_row.get("player_name", ""))
            ),
            "team": str(player_row.get("team", "")),
            "position": _positions_str(raw_positions),
            "is_pitcher": _is_pitcher(raw_positions),
            # overall_score is filled in below after the α-blend.
            "overall_score": scored["weighted_score"],
            "weighted_score": scored["weighted_score"],
            "talent_score": scored["talent_score"],
            "fit_score": scored["fit_score"],
            "category_scores": json.dumps(scored["category_scores"]),
            "recommended_drop_id": scored["recommended_drop_id"],
            "is_callup": is_callup,
            "days_since_callup": days_since,
        }

        # Carry over per-game display stats so the UI can render "Key Stats".
        for col in _DISPLAY_STAT_COLS:
            if col in player_row.index:
                rec[col] = player_row[col]

        records.append(rec)

    result = pd.DataFrame(records)
    if result.empty:
        return result

    # ── α-blend talent and fit into the final overall_score ───────────────
    alpha = alpha_from_season_progress(season_progress)
    result["overall_score"] = (
        alpha * result["talent_score"] + (1.0 - alpha) * result["fit_score"]
    )
    result = result.sort_values("overall_score", ascending=False).reset_index(drop=True)
    return result
