"""
matchup_analyzer.py

Pure functions for projecting week-end category totals and scoring matchups.
All functions accept DataFrames and return DataFrames — no DB or API calls.

Column contracts (from src/db/schema.py):
  fact_player_stats_daily: player_id, stat_date, ab, h, hr, sb, bb, errors, chances,
                           ip, w, k, walks_allowed, hits_allowed, sv, holds,
                           avg, ops, fpct, whip, k_bb, sv_h
  fact_projections: player_id, projection_date, target_week,
                    proj_ab, proj_h, proj_hr, proj_sb, proj_bb,
                    proj_ip, proj_w, proj_k, proj_sv_h,
                    proj_avg, proj_ops, proj_whip, proj_k_bb, proj_fpct,
                    proj_walks_allowed, proj_hits_allowed, proj_tb,
                    games_remaining, source
"""

from __future__ import annotations

import pandas as pd

# Counting stats that map directly to their projection columns.
_COUNTING_STAT_MAP: dict[str, str] = {
    "h": "proj_h",
    "hr": "proj_hr",
    "sb": "proj_sb",
    "bb": "proj_bb",
    "ab": "proj_ab",
    "hbp": "proj_hbp",
    "sf": "proj_sf",
    "tb": "proj_tb",
    "ip": "proj_ip",
    "w": "proj_w",
    "k": "proj_k",
    "walks_allowed": "proj_walks_allowed",
    "hits_allowed": "proj_hits_allowed",
    "sv": "proj_sv",
    "holds": "proj_holds",
    "errors": "proj_errors",
    "chances": "proj_chances",
}


def project_week_totals(
    stats_df: pd.DataFrame,
    projections_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine accumulated stats with remaining-week projections.

    Rate stats (AVG, OPS, FPCT, WHIP, K/BB) are computed from components,
    never averaged directly.

    Args:
        stats_df: Accumulated week-to-date stats. One row per player.
                  Must contain all fact_player_stats_daily columns.
        projections_df: Remaining-game projections. One row per player.
                        Must contain all fact_projections columns.

    Returns:
        DataFrame with one row per player and projected end-of-week totals.
        Includes all counting stat columns plus computed rate stats.
    """
    merged = stats_df.merge(projections_df, on="player_id", how="outer")

    # Fill NaN for numeric columns to 0 so arithmetic works cleanly.
    merged = merged.fillna(0)

    result = merged[["player_id"]].copy()

    # --- Aggregate counting stats ---
    counting_cols = [
        "h",
        "hr",
        "sb",
        "bb",
        "ab",
        "hbp",
        "sf",
        "tb",
        "ip",
        "w",
        "k",
        "walks_allowed",
        "hits_allowed",
        "sv",
        "holds",
        "errors",
        "chances",
    ]
    for col in counting_cols:
        proj_col = _COUNTING_STAT_MAP.get(col)
        stat_val = (
            merged[col] if col in merged.columns else pd.Series(0, index=merged.index)
        )
        if proj_col and proj_col in merged.columns:
            proj_val = merged[proj_col]
        else:
            proj_val = pd.Series(0, index=merged.index)
        result[col] = stat_val + proj_val

    # --- Compute rate stats from aggregated components ---

    # AVG = H / AB
    denom_ab = result["ab"]
    result["avg"] = (result["h"] / denom_ab).where(denom_ab > 0, 0.0)

    # OPS = OBP + SLG
    # OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
    # SLG = TB / AB
    obp_num = result["h"] + result["bb"] + result["hbp"]
    obp_denom = result["ab"] + result["bb"] + result["hbp"] + result["sf"]
    obp = (obp_num / obp_denom).where(obp_denom > 0, 0.0)
    slg = (result["tb"] / denom_ab).where(denom_ab > 0, 0.0)
    result["ops"] = obp + slg

    # FPCT = (chances - errors) / chances
    denom_chances = result["chances"]
    result["fpct"] = ((result["chances"] - result["errors"]) / denom_chances).where(
        denom_chances > 0, 0.0
    )

    # WHIP = (walks_allowed + hits_allowed) / IP
    denom_ip = result["ip"]
    result["whip"] = (
        (result["walks_allowed"] + result["hits_allowed"]) / denom_ip
    ).where(denom_ip > 0, 0.0)

    # K/BB = K / walks_allowed
    denom_walks = result["walks_allowed"]
    result["k_bb"] = (result["k"] / denom_walks).where(denom_walks > 0, 0.0)

    # SV_H = SV + holds
    result["sv_h"] = result["sv"] + result["holds"]

    return result


def score_categories(
    my_totals: pd.DataFrame,
    opp_totals: pd.DataFrame,
    category_config: dict[str, str],
) -> pd.DataFrame:
    """Score each category: who is winning and by how much.

    Args:
        my_totals: One-row DataFrame with my projected week totals.
        opp_totals: One-row DataFrame with opponent's projected week totals.
        category_config: Dict mapping category name → 'highest' or 'lowest'.
                         From LeagueSettings.category_win_direction.

    Returns:
        DataFrame with one row per category:
          category (str), my_value (float), opp_value (float),
          my_leads (bool), margin_pct (float), win_prob (float),
          status (str in {"safe_win","flippable_win","toss_up","flippable_loss","safe_loss"})

        For WHIP: lower is better — comparisons are inverted.
    """
    records = []

    my_row = my_totals.iloc[0]
    opp_row = opp_totals.iloc[0]

    for cat, direction in category_config.items():
        if cat not in my_row.index or cat not in opp_row.index:
            continue

        my_val = float(my_row[cat])
        opp_val = float(opp_row[cat])

        # Determine who leads based on direction
        if direction == "lowest":
            # Lower is better (WHIP)
            my_leads = my_val < opp_val
            # Compute margin as a fraction of opponent value (the reference)
            ref = opp_val if opp_val != 0 else (my_val if my_val != 0 else 1.0)
            if my_val == opp_val:
                margin_pct = 0.0
            else:
                margin_pct = abs(my_val - opp_val) / ref
        else:
            # Higher is better
            my_leads = my_val > opp_val
            ref = my_val if my_val != 0 else (opp_val if opp_val != 0 else 1.0)
            if my_val == opp_val:
                margin_pct = 0.0
            else:
                margin_pct = abs(my_val - opp_val) / ref

        # Classify status
        if my_val == opp_val:
            status = "toss_up"
            win_prob = 0.5
        elif margin_pct >= 0.15:
            status = "safe_win" if my_leads else "safe_loss"
            win_prob = 0.9 if my_leads else 0.1
        elif margin_pct >= 0.05:
            status = "flippable_win" if my_leads else "flippable_loss"
            win_prob = 0.7 if my_leads else 0.3
        else:
            status = "toss_up"
            win_prob = 0.55 if my_leads else 0.45

        records.append(
            {
                "category": cat,
                "my_value": my_val,
                "opp_value": opp_val,
                "my_leads": my_leads,
                "margin_pct": margin_pct,
                "win_prob": win_prob,
                "status": status,
            }
        )

    return pd.DataFrame(records)


def get_focus_categories(scored_df: pd.DataFrame) -> list[str]:
    """Return categories where lineup decisions still matter this week.

    Focus categories are those with status in:
    {"flippable_win", "toss_up", "flippable_loss"}

    Args:
        scored_df: Output of score_categories().

    Returns:
        List of category names (lowercase strings).
    """
    focus_statuses = {"flippable_win", "toss_up", "flippable_loss"}
    mask = scored_df["status"].isin(focus_statuses)
    return scored_df.loc[mask, "category"].tolist()


def check_ip_pace(
    my_stats_df: pd.DataFrame,
    days_remaining: int,
    min_ip: int = 21,
) -> dict[str, object]:
    """Check if pitching is on pace to meet the minimum IP requirement.

    Args:
        my_stats_df: Week-to-date stats for my roster. Must have 'ip' column.
        days_remaining: Days left in the scoring week (0–6).
        min_ip: Minimum innings pitched required per week (default 21).

    Returns:
        {
          "current_ip": float,
          "projected_ip": float,
          "min_ip": int,
          "on_pace": bool,
        }
    """
    current_ip = float(my_stats_df["ip"].sum()) if "ip" in my_stats_df.columns else 0.0

    # Days elapsed = total days in a 7-day week minus days remaining
    days_elapsed = 7 - days_remaining

    if days_elapsed <= 0:
        projected_ip = current_ip
    else:
        daily_rate = current_ip / days_elapsed
        projected_ip = current_ip + daily_rate * days_remaining

    on_pace = projected_ip >= min_ip

    return {
        "current_ip": current_ip,
        "projected_ip": projected_ip,
        "min_ip": min_ip,
        "on_pace": on_pace,
    }
