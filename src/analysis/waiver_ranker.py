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

# Positional need penalty: discount applied to overall_score when a player
# fills a position where the roster is already strong (all active slots filled
# with above-average performers). Prevents "you already have a great catcher"
# recommendations. The penalty is multiplicative: 1.0 = no penalty, 0.0 = full.
_POSITIONAL_SURPLUS_PENALTY = 0.35  # 65% penalty when position is stacked

# Positions that map to active roster slots (excludes Util, BN, IL, NA).
# Used to count how many slots exist per position.
_POSITION_SLOT_MAP: dict[str, list[str]] = {
    "C": ["C"],
    "1B": ["1B"],
    "2B": ["2B"],
    "3B": ["3B"],
    "SS": ["SS"],
    "OF": ["OF"],
    "SP": ["SP"],
    "RP": ["RP"],
    "P": ["P"],
}

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
    if isinstance(positions, np.ndarray):
        # Handle 0-d arrays (scalar wrapped in ndarray)
        if positions.ndim == 0:
            return str(positions).upper() in _PITCHER_POSITIONS
        return bool({str(p).upper() for p in positions} & _PITCHER_POSITIONS)
    if isinstance(positions, (list, tuple)):
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


def _get_positions_list(player: pd.Series[Any] | Any) -> list[str]:
    """Extract a flat list of position strings from a player row."""
    ep: Any = None
    if isinstance(player, pd.Series):
        if "eligible_positions" in player.index:
            ep = player["eligible_positions"]
        elif "positions" in player.index:
            ep = player["positions"]
    else:
        ep = player

    if ep is None:
        return []
    if isinstance(ep, np.ndarray):
        if ep.ndim == 0:
            return [str(ep).strip()]
        return [str(p).strip() for p in ep]
    if isinstance(ep, (list, tuple)):
        return [str(p).strip() for p in ep]
    if isinstance(ep, str):
        return [p.strip() for p in ep.replace("/", ",").split(",")]
    return []


# Single-slot active positions where losing your only eligible player
# creates a lineup hole that can't be filled. Multi-slot positions (OF × 3,
# Util × 2) are less scarce — losing one still leaves alternatives.
_SCARCE_POSITIONS: set[str] = {"C", "1B", "2B", "3B", "SS"}


def _is_sole_eligible(
    player_id: str,
    position: str,
    roster_df: pd.DataFrame,
    exclude_ids: set[str],
) -> bool:
    """Return True when *player_id* is the only roster player eligible at *position*.

    Checks the full roster (minus already-excluded players). If dropping this
    player would leave zero players who can fill the position, it's "sole eligible."
    """
    other = roster_df[
        (roster_df["player_id"].astype(str) != player_id)
        & (~roster_df["player_id"].astype(str).isin(exclude_ids))
    ]
    for _, row in other.iterrows():
        row_positions = _get_positions_list(row)
        if position in row_positions:
            return False
    return True


def find_recommended_drop(
    candidate_player: pd.Series[Any],
    my_roster_df: pd.DataFrame,
    config: LeagueSettings,
    exclude_ids: set[str] | None = None,
) -> str:
    """Find the best player to drop when adding the candidate.

    Uses a baseball-GM approach:
      - Enforce same player type (pitcher↔pitcher, hitter↔hitter).
      - Protect positionally scarce players (sole C, sole SS, etc.) unless the
        candidate can also play that position.
      - Prefer bench dead weight → bench → active at shared position → active.

    Priority:
      1. Same-type bench players with no games remaining (lowest score first).
      2. Same-type bench players (lowest score first).
      3. Same-type active players at shared positions (lowest score first).
      4. Same-type active players (lowest score, last-resort within type).

    At every tier, players who are the sole eligible at a scarce position
    (C/1B/2B/3B/SS) are protected — unless the FA candidate can also play
    that position, meaning the roster hole would be filled by the add.

    Args:
        candidate_player: The free agent being evaluated.
        my_roster_df: Current roster DataFrame.
        config: League settings (bench slot names).
        exclude_ids: Player IDs to skip (already used as drops this cycle).
    """
    if my_roster_df.empty:
        return ""

    if exclude_ids is None:
        exclude_ids = set()

    df = my_roster_df.copy()

    # Remove excluded players (already recommended as drops in this batch)
    if exclude_ids:
        df = df[~df["player_id"].astype(str).isin(exclude_ids)]
    if df.empty:
        return ""

    candidate_positions = _get_positions_list(candidate_player)
    candidate_is_pitcher = _is_pitcher(
        candidate_player.get(
            "eligible_positions", candidate_player.get("positions", [])
        )
    )

    # Filter to same player type — never drop a pitcher to add a hitter or vice versa
    def _row_is_pitcher(row: pd.Series[Any]) -> bool:
        return _is_pitcher(row.get("eligible_positions", row.get("positions", [])))

    type_mask = df.apply(_row_is_pitcher, axis=1)
    same_type_df = df[type_mask] if candidate_is_pitcher else df[~type_mask]

    # If no same-type players exist (unlikely), fall back to full roster
    if same_type_df.empty:
        same_type_df = df

    # ── Positional scarcity protection ────────────────────────────────────
    # Build a set of player_ids that are "protected" because they're the sole
    # eligible player at a scarce position, and the FA can't fill that slot.
    protected_ids: set[str] = set()
    for scarce_pos in _SCARCE_POSITIONS:
        # Count active roster slots for this position
        slot_count = config.active_positions.count(scarce_pos)
        if slot_count == 0:
            continue
        for _, row in same_type_df.iterrows():
            pid = str(row["player_id"])
            row_positions = _get_positions_list(row)
            if scarce_pos not in row_positions:
                continue
            # If the FA can also play this position, dropping this player is OK
            if scarce_pos in candidate_positions:
                continue
            # Check if this player is the sole eligible at this scarce position
            if _is_sole_eligible(pid, scarce_pos, my_roster_df, exclude_ids):
                protected_ids.add(pid)

    # Remove protected players from consideration
    droppable = same_type_df[~same_type_df["player_id"].astype(str).isin(protected_ids)]
    # If ALL same-type players are protected, there is no safe drop.
    # A GM would not create a lineup hole just to add a free agent.
    if droppable.empty:
        return ""

    bench_slots = set(config.bench_slots)

    def is_bench_only(row: pd.Series[Any]) -> bool:
        slot = str(row.get("slot", row.get("roster_slot", "")))
        return slot in bench_slots

    bench_candidates = droppable[droppable.apply(is_bench_only, axis=1)]
    active_candidates = droppable[~droppable.apply(is_bench_only, axis=1)]

    # Priority 1: Bench players with no games remaining
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

        # Priority 2: Bench players (lowest score)
        if "overall_score" in bench_candidates.columns:
            return str(
                bench_candidates.loc[
                    bench_candidates["overall_score"].idxmin(), "player_id"
                ]
            )
        return str(bench_candidates.iloc[0]["player_id"])

    # Priority 3: Active players sharing the candidate's position eligibility
    if candidate_positions:

        def shares_position(row: pd.Series[Any]) -> bool:
            row_positions = _get_positions_list(row)
            return bool(set(candidate_positions) & set(row_positions))

        pos_matches = active_candidates[
            active_candidates.apply(shares_position, axis=1)
        ]
        if not pos_matches.empty:
            if "overall_score" in pos_matches.columns:
                return str(
                    pos_matches.loc[pos_matches["overall_score"].idxmin(), "player_id"]
                )
            return str(pos_matches.iloc[0]["player_id"])

    # Priority 4: Any same-type droppable player (last resort)
    if not droppable.empty:
        if "overall_score" in droppable.columns:
            return str(droppable.loc[droppable["overall_score"].idxmin(), "player_id"])
        return str(droppable.iloc[0]["player_id"])

    return ""


def alpha_from_season_progress(progress: float) -> float:
    """Map season progress (0.0 - 1.0) to the talent/fit blend weight α.

    α decays linearly from ``_ALPHA_EARLY`` (start of season) to
    ``_ALPHA_LATE`` (end of season). Out-of-range inputs are clamped.
    """
    progress = max(0.0, min(1.0, float(progress)))
    return _ALPHA_EARLY - (_ALPHA_EARLY - _ALPHA_LATE) * progress


def _compute_positional_need(
    my_roster_df: pd.DataFrame, config: LeagueSettings
) -> dict[str, float]:
    """Compute a per-position need multiplier (1.0 = needed, penalized if stacked).

    For each position, counts the active roster slots allocated in the league
    and the number of quality players (above-median overall_score) already on
    the roster at that position. When a position is fully covered by strong
    players, returns a penalty multiplier; when there's a gap or weakness,
    returns 1.0 (no penalty).

    Pitchers are evaluated as a group (SP/RP/P slots combined) because pitcher
    fungibility is high in H2H categories.
    """
    if my_roster_df.empty:
        return {}

    # Count available active slots per position from league config
    active_slots = config.active_positions
    slot_counts: dict[str, int] = {}
    for slot in active_slots:
        slot_counts[slot] = slot_counts.get(slot, 0) + 1
    # Util slots can be filled by anyone — don't count them as position-specific
    slot_counts.pop("Util", None)

    # Determine roster quality at each position
    need: dict[str, float] = {}

    # Median overall_score across roster as quality threshold
    if "overall_score" not in my_roster_df.columns:
        return {}
    median_score = float(my_roster_df["overall_score"].median())

    for pos, n_slots in slot_counts.items():
        # Find roster players eligible at this position
        _pos = pos  # bind for lambda closure
        eligible_mask = my_roster_df.apply(
            lambda row, p=_pos: _player_eligible_at(row, p), axis=1
        )
        eligible = my_roster_df[eligible_mask]
        if eligible.empty:
            need[pos] = 1.0  # big need — no one at this position
            continue

        # Count how many are above-median quality (strictly above)
        quality_count = int((eligible["overall_score"] > median_score).sum())

        # If quality players exceed or meet the slot count, position is stacked
        if quality_count >= n_slots:
            need[pos] = _POSITIONAL_SURPLUS_PENALTY
        else:
            need[pos] = 1.0

    return need


def _player_eligible_at(row: pd.Series[Any], position: str) -> bool:
    """Check if a roster row is eligible at the given position."""
    ep = row.get("eligible_positions", row.get("positions", []))
    if isinstance(ep, np.ndarray):
        if ep.ndim == 0:
            return str(ep).upper() == position.upper()
        return position in [str(p) for p in ep]
    if isinstance(ep, (list, tuple)):
        return position in [str(p) for p in ep]
    if isinstance(ep, str):
        parts = [p.strip() for p in ep.replace("/", ",").split(",")]
        return position in parts
    return False


def _positional_need_multiplier(
    player_positions: Any,
    need_map: dict[str, float],
) -> float:
    """Return the best (highest) need multiplier across the player's positions.

    A multi-position player gets the benefit of their most needed position.
    E.g. a 2B/SS where SS is needed but 2B is stacked → gets 1.0.
    """
    if not need_map:
        return 1.0
    positions: list[str] = []
    if isinstance(player_positions, (list, tuple, np.ndarray)):
        positions = [str(p) for p in player_positions]
    elif isinstance(player_positions, str):
        positions = [p.strip() for p in player_positions.replace("/", ",").split(",")]
    if not positions:
        return 1.0
    # Pitcher positions are evaluated as a group
    multipliers = [need_map.get(p, 1.0) for p in positions]
    return max(multipliers) if multipliers else 1.0


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

    # ── Positional need adjustment ────────────────────────────────────────
    # Penalize FAs at positions where the roster is already stacked. This
    # prevents "add another catcher when you already have a great one."
    need_map = _compute_positional_need(my_roster_df, config)
    if need_map:
        result["_pos_multiplier"] = result["position"].apply(
            lambda p: _positional_need_multiplier(p, need_map)
        )
        result["overall_score"] = result["overall_score"] * result["_pos_multiplier"]
        result.drop(columns=["_pos_multiplier"], inplace=True)

    result = result.sort_values("overall_score", ascending=False).reset_index(drop=True)
    return result
