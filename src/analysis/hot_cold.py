"""
hot_cold.py

Hot / cold streak detection for hitters and pitchers using recent daily stats.

Hitter Streak Logic — ≥ 2 of 4 conditions → Hot / Cold
─────────────────────────────────────────────────────────
  HOT  (last 7 days of available data):
    1. Hit streak    : H > 0 on every one of the last 3 days with AB > 0
    2. 7-day AVG     : ≥ .320
    3. 7-day OPS     : ≥ .920
    4. Power / speed : ≥ 1 HR or ≥ 1 SB in the last 3 days

  COLD:
    1. Hitless run   : H = 0 on every one of the last 3 days with AB > 0
    2. 7-day AVG     : ≤ .180
    3. 7-day OPS     : ≤ .580
    4. No production : 0 HR and 0 SB in the last 7 days

Pitcher Streak Logic — rolling last 10 days (~2 starts) — ≥ 2 of 4
─────────────────────────────────────────────────────────────────────
  HOT (≥ 2 of 4 conditions):
    1. WHIP  : (hits_allowed + walks_allowed) / IP < 1.10
    2. RA9   : (hits_allowed + walks_allowed) × 9 / IP < 2.50
    3. K/9   : k × 9 / IP > 9.0
    4. K/BB  : k / walks_allowed > 3.0

  WARM (exactly 1 hot condition, 0 cold conditions):
    Label applied when a pitcher is pitching well but not dominant.

  COLD (≥ 2 of 4 conditions):
    1. WHIP  : > 1.60
    2. RA9   : > 5.00
    3. K/9   : < 6.0
    4. K/BB  : < 1.5

Labels: "🔥 Hot" | "☀️ Warm" | "❄️ Cold" | "—" (when insufficient data)
"""

from __future__ import annotations

import datetime

import pandas as pd

from src.analysis.shrinkage import (
    prior_ops_from_xwoba,
    prior_whip_from_xwoba_against,
)

_HOT = "🔥 Hot"
_WARM = "☀️ Warm"
_COLD = "❄️ Cold"
_NEUTRAL = "—"
_MIN_HITTER_DAYS = 3  # need at least 3 game-days to compute hitter streak
_MIN_PITCHER_IP = 1.0  # need at least 1 IP in window to compute pitcher streak

# ── Prior-based streak thresholds ──────────────────────────────────────────
# A continuous "is the player playing above/below his true-talent baseline
# right now?" signal. Compares rolling rate stats to the player's xwOBA-derived
# prior (or, if missing, season-to-date baseline). Far less noisy than the
# binary 4-of-4 condition test, and naturally calibrated per player.
_HITTER_OPS_HOT = 0.100  # 7-day OPS at least .100 above prior
_HITTER_OPS_COLD = -0.100
_HITTER_MIN_PA = 10
_PITCHER_WHIP_HOT = -0.30  # 10-day WHIP at least .30 below prior
_PITCHER_WHIP_COLD = 0.30
_PITCHER_MIN_IP_PRIOR = 5.0


def _hitter_streak(recent: pd.DataFrame) -> str:
    """Return hot/cold label for a hitter given their recent daily rows.

    Args:
        recent: Rows from fact_player_stats_daily, sorted oldest→newest,
                for a single player over the last 7 days.

    Returns:
        "🔥 Hot" | "❄️ Cold" | "—"
    """
    # Only rows where the player had an official at-bat (PA = AB > 0)
    at_bat_rows = recent[recent["ab"].fillna(0) > 0]
    if len(at_bat_rows) < _MIN_HITTER_DAYS:
        return _NEUTRAL

    # Aggregate 7-day totals
    h7 = int(recent["h"].fillna(0).sum())
    ab7 = int(recent["ab"].fillna(0).sum())
    tb7 = int(recent["tb"].fillna(0).sum())
    bb7 = int(recent["bb"].fillna(0).sum())
    hbp7 = int(recent["hbp"].fillna(0).sum())
    sf7 = int(recent["sf"].fillna(0).sum())
    hr7 = int(recent["hr"].fillna(0).sum())
    sb7 = int(recent["sb"].fillna(0).sum())

    avg7 = h7 / ab7 if ab7 > 0 else 0.0
    obp_denom = ab7 + bb7 + hbp7 + sf7
    obp7 = (h7 + bb7 + hbp7) / obp_denom if obp_denom > 0 else 0.0
    slg7 = tb7 / ab7 if ab7 > 0 else 0.0
    ops7 = obp7 + slg7

    # Last 3 game-days with at-bats
    last3 = at_bat_rows.tail(3)
    h3 = last3["h"].fillna(0).tolist()
    hr3 = int(last3["hr"].fillna(0).sum())
    sb3 = int(last3["sb"].fillna(0).sum())

    # Score HOT conditions
    hot_score = 0
    if all(h > 0 for h in h3):
        hot_score += 1
    if avg7 >= 0.320:
        hot_score += 1
    if ops7 >= 0.920:
        hot_score += 1
    if hr3 >= 1 or sb3 >= 1:
        hot_score += 1

    if hot_score >= 2:
        return _HOT

    # Score COLD conditions
    cold_score = 0
    if all(h == 0 for h in h3):
        cold_score += 1
    if avg7 <= 0.180:
        cold_score += 1
    if ops7 <= 0.580:
        cold_score += 1
    if hr7 == 0 and sb7 == 0:
        cold_score += 1

    if cold_score >= 2:
        return _COLD

    return _NEUTRAL


def _pitcher_streak(recent: pd.DataFrame) -> str:
    """Return hot/cold/warm label for a pitcher given their recent daily rows.

    Args:
        recent: Rows from fact_player_stats_daily, sorted oldest→newest,
                for a single player over the last 10 days.

    Returns:
        "🔥 Hot" | "☀️ Warm" | "❄️ Cold" | "—"
    """
    ip = float(recent["ip"].fillna(0).sum())
    if ip < _MIN_PITCHER_IP:
        return _NEUTRAL

    k = float(recent["k"].fillna(0).sum())
    wa = float(recent["walks_allowed"].fillna(0).sum())
    ha = float(recent["hits_allowed"].fillna(0).sum())

    whip = (ha + wa) / ip if ip > 0 else 99.0
    ra9 = (ha + wa) * 9 / ip if ip > 0 else 99.0
    k9 = k * 9 / ip if ip > 0 else 0.0
    kbb = k / wa if wa > 0 else (k if k > 0 else 0.0)

    hot_score = 0
    if whip < 1.10:
        hot_score += 1
    if ra9 < 2.50:
        hot_score += 1
    if k9 > 9.0:
        hot_score += 1
    if kbb > 3.0:
        hot_score += 1

    if hot_score >= 2:
        return _HOT

    cold_score = 0
    if whip > 1.60:
        cold_score += 1
    if ra9 > 5.00:
        cold_score += 1
    if k9 < 6.0:
        cold_score += 1
    if kbb < 1.5:
        cold_score += 1

    if cold_score >= 2:
        return _COLD

    if hot_score == 1:
        return _WARM

    return _NEUTRAL


def _hitter_streak_vs_prior(
    recent: pd.DataFrame,
    season_rows: pd.DataFrame,
    prior_ops: float | None,
) -> str:
    """Compare a hitter's rolling 7-day OPS to a true-talent prior.

    Falls back to the season-to-date OPS baseline (computed from
    ``season_rows``) when ``prior_ops`` is None.
    """
    ab7 = float(recent["ab"].fillna(0).sum())
    bb7 = float(recent["bb"].fillna(0).sum())
    hbp7 = float(recent["hbp"].fillna(0).sum())
    sf7 = float(recent["sf"].fillna(0).sum())

    pa7 = ab7 + bb7 + hbp7 + sf7
    if pa7 < _HITTER_MIN_PA:
        return _NEUTRAL

    h7 = float(recent["h"].fillna(0).sum())
    tb7 = float(recent["tb"].fillna(0).sum())

    obp_denom = ab7 + bb7 + hbp7 + sf7
    obp7 = (h7 + bb7 + hbp7) / obp_denom if obp_denom > 0 else 0.0
    slg7 = tb7 / ab7 if ab7 > 0 else 0.0
    ops7 = obp7 + slg7

    baseline = prior_ops
    if baseline is None:
        ab_s = float(season_rows["ab"].fillna(0).sum())
        if ab_s <= 0:
            return _NEUTRAL
        h_s = float(season_rows["h"].fillna(0).sum())
        bb_s = float(season_rows["bb"].fillna(0).sum())
        hbp_s = float(season_rows["hbp"].fillna(0).sum())
        sf_s = float(season_rows["sf"].fillna(0).sum())
        tb_s = float(season_rows["tb"].fillna(0).sum())
        obp_denom_s = ab_s + bb_s + hbp_s + sf_s
        obp_s = (h_s + bb_s + hbp_s) / obp_denom_s if obp_denom_s > 0 else 0.0
        slg_s = tb_s / ab_s if ab_s > 0 else 0.0
        baseline = obp_s + slg_s

    delta = ops7 - baseline
    if delta >= _HITTER_OPS_HOT:
        return _HOT
    if delta <= _HITTER_OPS_COLD:
        return _COLD
    return _NEUTRAL


def _pitcher_streak_vs_prior(
    recent: pd.DataFrame,
    season_rows: pd.DataFrame,
    prior_whip: float | None,
) -> str:
    """Compare a pitcher's rolling 10-day WHIP to a true-talent prior.

    Falls back to the season-to-date WHIP baseline when ``prior_whip`` is None.
    """
    ip_recent = float(recent["ip"].fillna(0).sum())
    if ip_recent < _MIN_PITCHER_IP:
        return _NEUTRAL

    ha = float(recent["hits_allowed"].fillna(0).sum())
    wa = float(recent["walks_allowed"].fillna(0).sum())
    whip_recent = (ha + wa) / ip_recent

    baseline = prior_whip
    if baseline is None:
        ip_s = float(season_rows["ip"].fillna(0).sum())
        if ip_s < _PITCHER_MIN_IP_PRIOR:
            return _NEUTRAL
        ha_s = float(season_rows["hits_allowed"].fillna(0).sum())
        wa_s = float(season_rows["walks_allowed"].fillna(0).sum())
        baseline = (ha_s + wa_s) / ip_s

    delta = whip_recent - baseline
    if delta <= _PITCHER_WHIP_HOT:
        return _HOT
    if delta >= _PITCHER_WHIP_COLD:
        return _COLD
    return _NEUTRAL


def streak_label(
    player_id: str,
    daily_df: pd.DataFrame,
    is_pitcher: bool,
    reference_date: datetime.date | None = None,
    advanced_df: pd.DataFrame | None = None,
) -> str:
    """Compute the hot/cold streak label for a single player.

    Args:
        player_id: The player's ID.
        daily_df: All rows from fact_player_stats_daily. Must contain
                  stat_date, ab, h, hr, sb, bb, hbp, sf, tb, ip, k,
                  walks_allowed, hits_allowed columns.
        is_pitcher: True if pitcher streak logic should be applied.
        reference_date: The "today" anchor for the rolling window.
                        Defaults to the max stat_date in daily_df.
        advanced_df: Optional fact_player_advanced_stats rows. When supplied,
            the streak label is computed as a continuous delta of the rolling
            rate stat (OPS for hitters, WHIP for pitchers) versus the player's
            xwOBA-derived true-talent prior. Falls back to season-to-date
            baseline if the player has no advanced row, and finally to the
            legacy 4-of-4 binary signal if no advanced data at all.

    Returns:
        "🔥 Hot" | "❄️ Cold" | "—"
    """
    player_rows = daily_df[daily_df["player_id"] == player_id].copy()
    if player_rows.empty:
        return _NEUTRAL

    player_rows["stat_date"] = pd.to_datetime(player_rows["stat_date"]).dt.date
    player_rows = player_rows.sort_values("stat_date")

    if reference_date is None:
        reference_date = player_rows["stat_date"].max()

    window_days = 10 if is_pitcher else 7
    cutoff = reference_date - datetime.timedelta(days=window_days)
    recent = player_rows[player_rows["stat_date"] > cutoff]

    if recent.empty:
        return _NEUTRAL

    # Prior-based path: compute rolling-rate vs true-talent delta.
    if advanced_df is not None:
        adv_row = advanced_df[advanced_df["player_id"].astype(str) == str(player_id)]
        if is_pitcher:
            xwoba_against = None
            if not adv_row.empty and "xwoba_against" in adv_row.columns:
                v = adv_row.iloc[0]["xwoba_against"]
                if pd.notna(v):
                    xwoba_against = float(v)
            return _pitcher_streak_vs_prior(
                recent, player_rows, prior_whip_from_xwoba_against(xwoba_against)
            )
        xwoba = None
        if not adv_row.empty and "xwoba" in adv_row.columns:
            v = adv_row.iloc[0]["xwoba"]
            if pd.notna(v):
                xwoba = float(v)
        return _hitter_streak_vs_prior(recent, player_rows, prior_ops_from_xwoba(xwoba))

    # Legacy 4-of-4 binary path (kept as a fallback for callers without
    # advanced stats — preserves existing behavior and tests).
    if is_pitcher:
        return _pitcher_streak(recent)
    return _hitter_streak(recent)


def annotate_with_streaks(
    df: pd.DataFrame,
    daily_df: pd.DataFrame,
    player_col: str = "player_id",
    position_col: str = "position",
    reference_date: datetime.date | None = None,
    advanced_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add a 'streak' column to a player DataFrame.

    A player is classified as a pitcher if their position contains 'SP', 'RP',
    or 'P' (and they have IP data in the daily stats).

    Args:
        df: Player DataFrame — must contain `player_col` and `position_col`.
        daily_df: fact_player_stats_daily rows.
        player_col: Column name holding player IDs.
        position_col: Column name holding position string(s).
        reference_date: Anchor date for rolling windows. Defaults to max
                        stat_date in daily_df.

    Returns:
        Copy of `df` with an added 'streak' column.
    """
    result = df.copy()

    def _is_pitcher(pos: str) -> bool:
        pos_upper = str(pos).upper()
        return any(p in pos_upper for p in ("SP", "RP", "/P"))

    result["streak"] = result.apply(
        lambda row: streak_label(
            player_id=str(row[player_col]),
            daily_df=daily_df,
            is_pitcher=_is_pitcher(str(row.get(position_col, ""))),
            reference_date=reference_date,
            advanced_df=advanced_df,
        ),
        axis=1,
    )
    return result


def match_win_probability(win_probs: list[float]) -> float:
    """Probability of winning more than half of n categories.

    Uses exact Poisson binomial DP: treats each category as an independent
    Bernoulli trial with its own win probability, then computes P(X > n/2).

    Args:
        win_probs: Per-category win probabilities, each in [0, 1].

    Returns:
        Probability in [0, 1] of winning the overall matchup.
    """
    n = len(win_probs)
    if n == 0:
        return 0.0

    # dp[j] = P(exactly j category wins from categories seen so far)
    dp = [0.0] * (n + 1)
    dp[0] = 1.0
    for p in win_probs:
        new_dp = [0.0] * (n + 1)
        for j in range(n + 1):
            if dp[j] == 0.0:
                continue
            new_dp[j] += dp[j] * (1.0 - p)
            if j < n:
                new_dp[j + 1] += dp[j] * p
        dp = new_dp

    threshold = n // 2  # must win strictly more than half
    return sum(dp[j] for j in range(threshold + 1, n + 1))
