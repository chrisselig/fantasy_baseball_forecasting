"""
shrinkage.py

Empirical-Bayes shrinkage for early-season rate stats.

In the first weeks of the season, observed rate stats (AVG, OPS, HR/PA, K/9,
WHIP, …) are extremely noisy: a hitter who goes 6-for-12 looks like a .500
hitter, a pitcher with 5 IP and 2 ER looks like he has a 3.60 ERA. The
underlying *true talent* is much closer to the priors we already know about
the player from Statcast / projections.

This module applies a simple Bayesian-shrinkage estimator:

    shrunk = (n × observed + k × prior) / (n + k)

where ``n`` is the number of observed denominator events (PA, AB, BF, BBE),
``k`` is a stabilization constant (the number of events at which the stat
becomes about as informative as the prior — Russell Carleton's well-known
"reliability" constants), and ``prior`` is a Statcast-derived expectation
for that player.

Statcast → outcome priors
=========================
We use empirically calibrated linear approximations:

  OPS                  ≈ 2.86 × xwOBA  − 0.18
  AVG                  ≈ 0.625 × xwOBA + 0.07
  HR per plate appearance ≈ Barrel% (decimal) × 0.006
  WHIP                 ≈ 5.7 × xwOBA-against − 0.50
  K per batter faced   ≈ K-BB% (decimal) + 0.13      (rough; floor at 0.15)

These map "what should have happened" (Statcast quality of contact) into the
fantasy stat lines we score. They are intentionally simple and league-average
calibrated; the shrinkage layer only needs them to be roughly unbiased — the
weighting math takes care of the rest.

Stabilization constants (Carleton, FanGraphs)
============================================
  HR/PA             ~ 170 PA
  AVG (H/AB)        ~ 910 AB
  OPS (proxy)       ~ 460 PA
  Barrel%           ~  50 BBE
  K%  (hitter)      ~  60 PA
  BB% (hitter)      ~ 120 PA
  K%  (pitcher)     ~  70 BF
  BB% (pitcher)     ~ 170 BF
  WHIP              ~ 770 BF

Public API
==========
  shrink_rate(observed, n_obs, prior, n_stab) -> float
  prior_ops_from_xwoba(xwoba) -> float
  prior_avg_from_xwoba(xwoba) -> float
  prior_hr_per_pa_from_barrel(barrel_pct) -> float
  prior_whip_from_xwoba_against(xwoba_against) -> float
  prior_k_per_bf_from_kbb(k_bb_pct) -> float
  apply_shrinkage_to_rates(rates_df, advanced_df) -> pd.DataFrame
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# ── Stabilization constants (Russell Carleton) ─────────────────────────────
HR_PA_K: float = 170.0
AVG_AB_K: float = 910.0
OPS_PA_K: float = 460.0
WHIP_BF_K: float = 770.0
K_BF_K: float = 70.0

# Approximate plate appearances per game played (used when we only know G).
PA_PER_GAME: float = 4.3
# Approximate batters faced per pitching appearance.
BF_PER_GAME: float = 4.3 * 9  # ≈ 38.7 — fine for shrinkage weighting


def shrink_rate(
    observed: float,
    n_obs: float,
    prior: float,
    n_stab: float,
) -> float:
    """Bayesian shrinkage of an observed rate toward a prior.

    Args:
        observed: The observed rate (e.g. .520 OPS through 30 PA).
        n_obs: The denominator events backing ``observed`` (e.g. 30 PA).
        prior: The prior rate (e.g. .720 OPS implied by xwOBA).
        n_stab: Stabilization constant — the n at which observed and prior
            should receive equal weight.

    Returns:
        The shrunk rate. If both ``n_obs`` and ``n_stab`` are zero, returns
        ``observed`` unchanged.
    """
    if observed is None or (isinstance(observed, float) and math.isnan(observed)):
        observed = 0.0
    if prior is None or (isinstance(prior, float) and math.isnan(prior)):
        # Without a prior, fall back to the raw observation.
        return float(observed)
    n_obs = max(0.0, float(n_obs))
    n_stab = max(0.0, float(n_stab))
    denom = n_obs + n_stab
    if denom <= 0.0:
        return float(observed)
    return (n_obs * float(observed) + n_stab * float(prior)) / denom


# ── Statcast → outcome priors ──────────────────────────────────────────────


def prior_ops_from_xwoba(xwoba: float | None) -> float | None:
    """Linear OPS prior derived from xwOBA. Returns None if xwOBA is missing."""
    if xwoba is None or (isinstance(xwoba, float) and math.isnan(xwoba)):
        return None
    return max(0.0, 2.86 * float(xwoba) - 0.18)


def prior_avg_from_xwoba(xwoba: float | None) -> float | None:
    """Linear AVG prior derived from xwOBA."""
    if xwoba is None or (isinstance(xwoba, float) and math.isnan(xwoba)):
        return None
    return max(0.0, 0.625 * float(xwoba) + 0.07)


def prior_hr_per_pa_from_barrel(barrel_pct: float | None) -> float | None:
    """HR-per-PA prior from Barrel% (entered as a percentage 0-100)."""
    if barrel_pct is None or (isinstance(barrel_pct, float) and math.isnan(barrel_pct)):
        return None
    # Barrels happen on contact, ~75% of PAs end in a BBE; barrels become HR
    # ~80% of the time. (barrel_pct/100) × 0.75 × 0.8 ≈ barrel_pct × 0.006.
    return max(0.0, float(barrel_pct) * 0.006)


def prior_whip_from_xwoba_against(xwoba_against: float | None) -> float | None:
    """WHIP prior from opponent xwOBA. Floored at 0.7 for plausibility."""
    if xwoba_against is None or (
        isinstance(xwoba_against, float) and math.isnan(xwoba_against)
    ):
        return None
    return max(0.7, 5.7 * float(xwoba_against) - 0.5)


def prior_k_per_bf_from_kbb(k_bb_pct: float | None) -> float | None:
    """K-per-BF prior from K-BB% (entered as a percentage 0-100).

    K-BB% itself is a strong stabilizing signal of strikeout rate. We add a
    league-average BB-rate (~8%) back to recover an approximate K-rate.
    """
    if k_bb_pct is None or (isinstance(k_bb_pct, float) and math.isnan(k_bb_pct)):
        return None
    return max(0.10, float(k_bb_pct) / 100.0 + 0.08)


# ── DataFrame helper ───────────────────────────────────────────────────────

# Columns the helper expects on rates_df, populated upstream by
# `_query_player_rates` in src/pipeline/daily_run.py.
_RATE_COLS = ("avg", "ops", "hr", "whip", "k")
_PA_PROXY_COL = "games_played"


def apply_shrinkage_to_rates(
    rates_df: pd.DataFrame,
    advanced_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Return ``rates_df`` with key per-game rates shrunk toward Statcast priors.

    Only the columns we have priors for are touched: ``avg``, ``ops``, ``hr``,
    ``whip``, ``k``. The shape and other columns are preserved so callers
    (the waiver ranker) can keep operating on the same DataFrame.

    Args:
        rates_df: Output of ``_query_player_rates`` — has ``player_id``,
            ``games_played``, and per-game rate columns.
        advanced_df: ``fact_player_advanced_stats`` rows for the relevant
            players (current season). May be ``None`` or empty, in which case
            no shrinkage is applied.

    Returns:
        A new DataFrame with shrunk rate columns.
    """
    if rates_df is None or rates_df.empty:
        return rates_df
    out = rates_df.copy()
    if advanced_df is None or advanced_df.empty:
        return out

    adv = advanced_df.copy()
    adv["player_id"] = adv["player_id"].astype(str)
    out["player_id"] = out["player_id"].astype(str)
    adv_idx = adv.set_index("player_id")

    if _PA_PROXY_COL not in out.columns:
        out[_PA_PROXY_COL] = 0
    games_arr = (
        pd.to_numeric(out[_PA_PROXY_COL], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=float)
    )

    # Ensure target columns exist (zeros for missing).
    for col in _RATE_COLS:
        if col not in out.columns:
            out[col] = 0.0

    def _adv_value(pid: str, col: str) -> float | None:
        if pid not in adv_idx.index or col not in adv_idx.columns:
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

    shrunk_avg: list[float] = []
    shrunk_ops: list[float] = []
    shrunk_hr: list[float] = []
    shrunk_whip: list[float] = []
    shrunk_k: list[float] = []

    for pos, (_, row) in enumerate(out.iterrows()):
        pid = str(row["player_id"])
        n_pa = float(games_arr[pos]) * PA_PER_GAME
        n_bf = float(games_arr[pos]) * BF_PER_GAME

        # ── Hitter side ────────────────────────────────────────────────
        xwoba = _adv_value(pid, "xwoba")
        barrel = _adv_value(pid, "barrel_pct")

        avg_obs = float(row.get("avg", 0.0) or 0.0)
        ops_obs = float(row.get("ops", 0.0) or 0.0)
        hr_obs_pg = float(row.get("hr", 0.0) or 0.0)

        avg_prior = prior_avg_from_xwoba(xwoba)
        ops_prior = prior_ops_from_xwoba(xwoba)
        hr_prior_per_pa = prior_hr_per_pa_from_barrel(barrel)

        if avg_prior is not None and n_pa > 0:
            shrunk_avg.append(shrink_rate(avg_obs, n_pa, avg_prior, AVG_AB_K))
        else:
            shrunk_avg.append(avg_obs)

        if ops_prior is not None and n_pa > 0:
            shrunk_ops.append(shrink_rate(ops_obs, n_pa, ops_prior, OPS_PA_K))
        else:
            shrunk_ops.append(ops_obs)

        if hr_prior_per_pa is not None and n_pa > 0:
            # Compare apples to apples in PA-space, then convert back to per-game.
            hr_obs_per_pa = hr_obs_pg / PA_PER_GAME
            shrunk_per_pa = shrink_rate(hr_obs_per_pa, n_pa, hr_prior_per_pa, HR_PA_K)
            shrunk_hr.append(shrunk_per_pa * PA_PER_GAME)
        else:
            shrunk_hr.append(hr_obs_pg)

        # ── Pitcher side ───────────────────────────────────────────────
        xwoba_against = _adv_value(pid, "xwoba_against")
        k_bb_pct = _adv_value(pid, "k_bb_pct")

        whip_obs = float(row.get("whip", 0.0) or 0.0)
        k_obs_pg = float(row.get("k", 0.0) or 0.0)

        whip_prior = prior_whip_from_xwoba_against(xwoba_against)
        k_prior_per_bf = prior_k_per_bf_from_kbb(k_bb_pct)

        if whip_prior is not None and n_bf > 0:
            shrunk_whip.append(shrink_rate(whip_obs, n_bf, whip_prior, WHIP_BF_K))
        else:
            shrunk_whip.append(whip_obs)

        if k_prior_per_bf is not None and n_bf > 0:
            k_obs_per_bf = k_obs_pg / BF_PER_GAME
            shrunk_per_bf = shrink_rate(k_obs_per_bf, n_bf, k_prior_per_bf, K_BF_K)
            shrunk_k.append(shrunk_per_bf * BF_PER_GAME)
        else:
            shrunk_k.append(k_obs_pg)

    out["avg"] = np.array(shrunk_avg, dtype=float)
    out["ops"] = np.array(shrunk_ops, dtype=float)
    out["hr"] = np.array(shrunk_hr, dtype=float)
    out["whip"] = np.array(shrunk_whip, dtype=float)
    out["k"] = np.array(shrunk_k, dtype=float)
    return out
