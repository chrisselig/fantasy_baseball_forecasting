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

import math

import pandas as pd

from src.analysis.shrinkage import (
    BF_PER_GAME,
    HR_PA_K,
    K_BF_K,
    PA_PER_GAME,
    WHIP_BF_K,
    prior_hr_per_pa_from_barrel,
    prior_k_per_bf_from_kbb,
    prior_whip_from_xwoba_against,
    shrink_rate,
)

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


def _shrink_projection_rates(
    projections_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    advanced_df: pd.DataFrame,
) -> pd.DataFrame:
    """Shrink per-game projection rates toward Statcast-implied priors.

    The pipeline's projections are season-to-date pace divided by games
    played, which is extremely noisy in the first weeks of a season. We
    apply Bayesian shrinkage to the per-game projection components for the
    rate-driving stats (HR, K, walks/hits allowed) using each player's
    Statcast prior. The shrinkage weight is the player's *accumulated*
    plate appearances / batters faced, taken from ``stats_df`` — so the
    longer a player has been hot, the less his projection is regressed.

    The columns we touch are:
        proj_hr             — shrunk via Barrel%-derived HR/PA prior
        proj_k              — shrunk via K-BB%-derived K/BF prior
        proj_walks_allowed  — shrunk via xwOBA-against-derived WHIP prior
        proj_hits_allowed   — shrunk via xwOBA-against-derived WHIP prior

    All other projection columns are left unchanged so the existing
    aggregation math (AVG/OPS/FPCT recomputation) keeps working.
    """
    if projections_df is None or projections_df.empty:
        return projections_df
    if advanced_df is None or advanced_df.empty:
        return projections_df

    out = projections_df.copy()
    out["player_id"] = out["player_id"].astype(str)

    adv = advanced_df.copy()
    adv["player_id"] = adv["player_id"].astype(str)
    adv_idx = adv.set_index("player_id")

    stats = stats_df.copy() if stats_df is not None else pd.DataFrame()
    if not stats.empty and "player_id" in stats.columns:
        stats["player_id"] = stats["player_id"].astype(str)
        stats_idx = stats.set_index("player_id")
    else:
        stats_idx = pd.DataFrame()

    def _adv(pid: str, col: str) -> float | None:
        if stats_idx is None or pid not in adv_idx.index or col not in adv_idx.columns:
            return None
        raw = adv_idx.at[pid, col]
        if isinstance(raw, pd.Series):
            raw = raw.iloc[0]
        if raw is None:
            return None
        try:
            val = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if math.isnan(val):
            return None
        return val

    def _pa_observed(pid: str) -> float:
        if not isinstance(stats_idx, pd.DataFrame) or pid not in stats_idx.index:
            return 0.0
        row = stats_idx.loc[pid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        ab = float(row.get("ab", 0) or 0)
        bb = float(row.get("bb", 0) or 0)
        hbp = float(row.get("hbp", 0) or 0)
        sf = float(row.get("sf", 0) or 0)
        return ab + bb + hbp + sf

    def _bf_observed(pid: str) -> float:
        if not isinstance(stats_idx, pd.DataFrame) or pid not in stats_idx.index:
            return 0.0
        row = stats_idx.loc[pid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        ip = float(row.get("ip", 0) or 0)
        # Approx BF: ip * 4.3 if no walks_allowed/hits_allowed available.
        ha = float(row.get("hits_allowed", 0) or 0)
        wa = float(row.get("walks_allowed", 0) or 0)
        k = float(row.get("k", 0) or 0)
        outs = ip * 3
        return outs + ha + wa + k * 0  # k already in outs; keep simple

    # Iterate row-wise — projections are typically <500 rows so this is fine.
    new_hr: list[float] = []
    new_k: list[float] = []
    new_wa: list[float] = []
    new_ha: list[float] = []
    for _, row in out.iterrows():
        pid = str(row["player_id"])
        n_pa = _pa_observed(pid)
        n_bf = _bf_observed(pid)

        # ── Hitter HR/game ────────────────────────────────────────────
        proj_hr_pg = float(row.get("proj_hr", 0.0) or 0.0)
        barrel = _adv(pid, "barrel_pct")
        hr_prior_pa = prior_hr_per_pa_from_barrel(barrel)
        if hr_prior_pa is not None and n_pa > 0:
            obs_per_pa = proj_hr_pg / PA_PER_GAME
            shrunk = shrink_rate(obs_per_pa, n_pa, hr_prior_pa, HR_PA_K)
            new_hr.append(shrunk * PA_PER_GAME)
        else:
            new_hr.append(proj_hr_pg)

        # ── Pitcher K/game ────────────────────────────────────────────
        proj_k_pg = float(row.get("proj_k", 0.0) or 0.0)
        kbb_pct = _adv(pid, "k_bb_pct")
        k_prior_bf = prior_k_per_bf_from_kbb(kbb_pct)
        if k_prior_bf is not None and n_bf > 0:
            obs_per_bf = proj_k_pg / BF_PER_GAME
            shrunk_k = shrink_rate(obs_per_bf, n_bf, k_prior_bf, K_BF_K)
            new_k.append(shrunk_k * BF_PER_GAME)
        else:
            new_k.append(proj_k_pg)

        # ── Pitcher walks_allowed + hits_allowed (drives WHIP) ────────
        proj_wa = float(row.get("proj_walks_allowed", 0.0) or 0.0)
        proj_ha = float(row.get("proj_hits_allowed", 0.0) or 0.0)
        xwoba_against = _adv(pid, "xwoba_against")
        whip_prior = prior_whip_from_xwoba_against(xwoba_against)
        if whip_prior is not None and n_bf > 0:
            # Shrink the combined per-game (W+H) toward the prior, then
            # split back proportionally between walks and hits.
            combined_pg = proj_wa + proj_ha
            obs_per_bf = combined_pg / BF_PER_GAME
            # Convert WHIP prior to per-BF: WHIP = (W+H)/IP, IP ≈ BF/4.3.
            prior_per_bf = whip_prior / 4.3
            shrunk_pb = shrink_rate(obs_per_bf, n_bf, prior_per_bf, WHIP_BF_K)
            shrunk_combined = shrunk_pb * BF_PER_GAME
            split = proj_wa / combined_pg if combined_pg > 0 else 0.5
            new_wa.append(shrunk_combined * split)
            new_ha.append(shrunk_combined * (1.0 - split))
        else:
            new_wa.append(proj_wa)
            new_ha.append(proj_ha)

    out["proj_hr"] = new_hr
    out["proj_k"] = new_k
    out["proj_walks_allowed"] = new_wa
    out["proj_hits_allowed"] = new_ha
    return out


def project_week_totals(
    stats_df: pd.DataFrame,
    projections_df: pd.DataFrame,
    days_remaining: int = 0,
    advanced_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine accumulated stats with remaining-week projections.

    Per-game projection rates are scaled by each player's expected remaining
    games: ``per_game_rate × games_per_day × days_remaining``.  The
    ``games_per_day`` frequency factor (season games / season days) ensures
    that SPs who pitch every 5th day project ~1 more start, not 5.
    Rate stats (AVG, OPS, FPCT, WHIP, K/BB) are recomputed from the
    combined components, never averaged directly.

    Args:
        stats_df: Accumulated week-to-date stats. One row per player.
                  Must contain all fact_player_stats_daily columns.
        projections_df: Remaining-week projections (per-game rates).
                        One row per player.  Must contain all
                        fact_projections columns.  May include a
                        ``games_per_day`` column for frequency scaling.
        days_remaining: Days left in the matchup week (0–6).  Per-game
                        projection rates are multiplied by
                        ``games_per_day × days_remaining``.
        advanced_df: Optional fact_player_advanced_stats rows. When supplied,
            per-game projection rates for HR, K, walks_allowed, and
            hits_allowed are first shrunk toward Statcast-implied priors,
            using each player's accumulated PA / BF as the shrinkage weight.
            This eliminates the wild early-season pace projections that the
            naive ``actuals + per_game × days`` math otherwise produces.

    Returns:
        DataFrame with one row per player and projected end-of-week totals.
        Includes all counting stat columns plus computed rate stats.
    """
    if advanced_df is not None and not advanced_df.empty:
        projections_df = _shrink_projection_rates(projections_df, stats_df, advanced_df)
    merged = stats_df.merge(projections_df, on="player_id", how="outer")

    # Fill NaN for numeric columns to 0 so arithmetic works cleanly.
    merged = merged.fillna(0)

    result = merged[["player_id"]].copy()

    # Per-player scale: per-game rate × games_per_day × days_remaining.
    # games_per_day captures how often each player actually appears
    # (e.g. ~0.2 for SPs, ~0.85 for everyday hitters). Falls back to 1.0
    # (legacy behavior) when the column is absent.
    days = max(days_remaining, 0)
    if "games_per_day" in merged.columns:
        freq = merged["games_per_day"].fillna(1.0).clip(lower=0.0, upper=1.0)
    else:
        freq = pd.Series(1.0, index=merged.index)
    scale = freq * days

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
            proj_val = merged[proj_col] * scale
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
    result: list[str] = scored_df.loc[mask, "category"].tolist()
    return result


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
