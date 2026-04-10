"""
tests/analysis/test_shrinkage.py

Unit tests for src/analysis/shrinkage.py.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.analysis.shrinkage import (
    AVG_AB_K,
    HR_PA_K,
    OPS_PA_K,
    PA_PER_GAME,
    WHIP_BF_K,
    apply_shrinkage_to_rates,
    prior_avg_from_xwoba,
    prior_hr_per_pa_from_barrel,
    prior_k_per_bf_from_kbb,
    prior_ops_from_xwoba,
    prior_whip_from_xwoba_against,
    shrink_rate,
)


class TestShrinkRate:
    def test_zero_obs_returns_prior(self) -> None:
        assert shrink_rate(0.0, 0, 0.750, 460) == pytest.approx(0.750)

    def test_huge_obs_returns_observed(self) -> None:
        # With 10,000 PA the stabilization barely moves the estimate.
        shrunk = shrink_rate(0.900, 10_000, 0.700, 460)
        assert shrunk == pytest.approx(0.900, abs=0.01)

    def test_equal_n_is_midpoint(self) -> None:
        # At n_obs == n_stab, shrunk is exactly the midpoint.
        shrunk = shrink_rate(1.000, 460, 0.700, 460)
        assert shrunk == pytest.approx(0.850)

    def test_nan_observed_treated_as_zero(self) -> None:
        shrunk = shrink_rate(float("nan"), 100, 0.700, 460)
        # (100 × 0 + 460 × 0.7) / 560
        assert shrunk == pytest.approx((460 * 0.7) / 560)

    def test_missing_prior_returns_observed(self) -> None:
        assert shrink_rate(0.800, 50, None, 460) == pytest.approx(0.800)  # type: ignore[arg-type]

    def test_zero_denominator_returns_observed(self) -> None:
        assert shrink_rate(0.400, 0, 0.700, 0) == pytest.approx(0.400)


class TestPriorHelpers:
    def test_prior_ops_from_xwoba_reasonable(self) -> None:
        # .320 xwOBA ≈ league average → OPS ~.735
        assert prior_ops_from_xwoba(0.320) == pytest.approx(0.7352, abs=0.001)

    def test_prior_avg_from_xwoba_reasonable(self) -> None:
        assert prior_avg_from_xwoba(0.320) == pytest.approx(0.27, abs=0.005)

    def test_prior_hr_per_pa_from_barrel(self) -> None:
        # 10% barrel → ~6% HR/PA (elite power)
        assert prior_hr_per_pa_from_barrel(10.0) == pytest.approx(0.06, abs=0.001)

    def test_prior_whip_from_xwoba_against(self) -> None:
        # .300 xwOBA-against → WHIP ~1.21
        assert prior_whip_from_xwoba_against(0.300) == pytest.approx(1.21, abs=0.01)

    def test_prior_k_per_bf_from_kbb(self) -> None:
        # 20% K-BB% → ~28% K rate (20 + 8)
        assert prior_k_per_bf_from_kbb(20.0) == pytest.approx(0.28, abs=0.005)

    def test_priors_return_none_for_missing(self) -> None:
        assert prior_ops_from_xwoba(None) is None
        assert prior_avg_from_xwoba(float("nan")) is None
        assert prior_hr_per_pa_from_barrel(None) is None
        assert prior_whip_from_xwoba_against(None) is None
        assert prior_k_per_bf_from_kbb(None) is None


class TestApplyShrinkageToRates:
    def test_empty_rates_passes_through(self) -> None:
        df = pd.DataFrame(columns=["player_id", "games_played", "avg", "ops"])
        out = apply_shrinkage_to_rates(df, pd.DataFrame())
        assert out.empty

    def test_none_advanced_returns_unchanged(self) -> None:
        rates = pd.DataFrame(
            [
                {
                    "player_id": "p1",
                    "games_played": 10,
                    "avg": 0.4,
                    "ops": 1.2,
                    "hr": 0.5,
                    "whip": 0.8,
                    "k": 6.0,
                }
            ]
        )
        out = apply_shrinkage_to_rates(rates, None)
        assert out.iloc[0]["avg"] == pytest.approx(0.4)
        assert out.iloc[0]["ops"] == pytest.approx(1.2)

    def test_hot_hitter_regressed_toward_prior(self) -> None:
        # Hitter with .500 AVG and 1.400 OPS on 10 games (avg xwOBA=.320)
        # should be strongly pulled toward league-average.
        rates = pd.DataFrame(
            [
                {
                    "player_id": "p1",
                    "games_played": 10,
                    "avg": 0.500,
                    "ops": 1.400,
                    "hr": 1.0,
                    "whip": 0.0,
                    "k": 0.0,
                }
            ]
        )
        advanced = pd.DataFrame(
            [
                {
                    "player_id": "p1",
                    "xwoba": 0.320,
                    "barrel_pct": 8.0,
                    "xwoba_against": None,
                    "k_bb_pct": None,
                }
            ]
        )
        out = apply_shrinkage_to_rates(rates, advanced)
        row = out.iloc[0]
        # Observed 1.400, prior ~.735, n_pa = 10*4.3=43, k=460 → shrunk much closer to prior
        expected_ops = (43 * 1.400 + OPS_PA_K * 0.7352) / (43 + OPS_PA_K)
        assert row["ops"] == pytest.approx(expected_ops, abs=0.01)
        assert row["ops"] < 0.850  # was 1.400, now regressed hard
        # AVG should also be pulled down
        expected_avg = (43 * 0.500 + AVG_AB_K * 0.27) / (43 + AVG_AB_K)
        assert row["avg"] == pytest.approx(expected_avg, abs=0.01)

    def test_hr_shrunk_in_pa_space(self) -> None:
        rates = pd.DataFrame(
            [
                {
                    "player_id": "p1",
                    "games_played": 5,
                    "avg": 0.25,
                    "ops": 0.7,
                    "hr": 2.0,
                    "whip": 0.0,
                    "k": 0.0,
                }
            ]
        )
        advanced = pd.DataFrame(
            [
                {
                    "player_id": "p1",
                    "xwoba": 0.310,
                    "barrel_pct": 5.0,
                    "xwoba_against": None,
                    "k_bb_pct": None,
                }
            ]
        )
        out = apply_shrinkage_to_rates(rates, advanced)
        row = out.iloc[0]
        n_pa = 5 * PA_PER_GAME
        prior = 5.0 * 0.006  # 0.03 per PA
        obs_per_pa = 2.0 / PA_PER_GAME
        expected_per_pa = (n_pa * obs_per_pa + HR_PA_K * prior) / (n_pa + HR_PA_K)
        assert row["hr"] == pytest.approx(expected_per_pa * PA_PER_GAME, abs=0.001)

    def test_pitcher_whip_shrinkage(self) -> None:
        rates = pd.DataFrame(
            [
                {
                    "player_id": "p1",
                    "games_played": 2,
                    "avg": 0.0,
                    "ops": 0.0,
                    "hr": 0.0,
                    "whip": 2.50,
                    "k": 8.0,
                }
            ]
        )
        advanced = pd.DataFrame(
            [
                {
                    "player_id": "p1",
                    "xwoba": None,
                    "barrel_pct": None,
                    "xwoba_against": 0.300,
                    "k_bb_pct": 15.0,
                }
            ]
        )
        out = apply_shrinkage_to_rates(rates, advanced)
        row = out.iloc[0]
        n_bf = 2 * 4.3 * 9
        prior_whip = 5.7 * 0.300 - 0.5  # 1.21
        expected_whip = (n_bf * 2.50 + WHIP_BF_K * prior_whip) / (n_bf + WHIP_BF_K)
        assert row["whip"] == pytest.approx(expected_whip, abs=0.01)
        assert row["whip"] < 2.50  # regressed down from 2.50
        assert row["whip"] > 1.21  # but not all the way to prior

    def test_missing_advanced_for_player_preserves_observed(self) -> None:
        rates = pd.DataFrame(
            [
                {
                    "player_id": "p1",
                    "games_played": 10,
                    "avg": 0.333,
                    "ops": 0.900,
                    "hr": 0.5,
                    "whip": 0.0,
                    "k": 0.0,
                }
            ]
        )
        # Advanced table has a different player
        advanced = pd.DataFrame(
            [
                {
                    "player_id": "p2",
                    "xwoba": 0.320,
                    "barrel_pct": 8.0,
                    "xwoba_against": None,
                    "k_bb_pct": None,
                }
            ]
        )
        out = apply_shrinkage_to_rates(rates, advanced)
        row = out.iloc[0]
        assert row["avg"] == pytest.approx(0.333)
        assert row["ops"] == pytest.approx(0.900)
        assert row["hr"] == pytest.approx(0.5)

    def test_zero_games_leaves_values_unchanged(self) -> None:
        rates = pd.DataFrame(
            [
                {
                    "player_id": "p1",
                    "games_played": 0,
                    "avg": 0.0,
                    "ops": 0.0,
                    "hr": 0.0,
                    "whip": 0.0,
                    "k": 0.0,
                }
            ]
        )
        advanced = pd.DataFrame(
            [
                {
                    "player_id": "p1",
                    "xwoba": 0.320,
                    "barrel_pct": 8.0,
                    "xwoba_against": 0.300,
                    "k_bb_pct": 15.0,
                }
            ]
        )
        out = apply_shrinkage_to_rates(rates, advanced)
        # n_pa = 0 → all values untouched (still 0.0)
        row = out.iloc[0]
        assert row["ops"] == pytest.approx(0.0)
        assert row["whip"] == pytest.approx(0.0)

    def test_nan_advanced_values_handled(self) -> None:
        rates = pd.DataFrame(
            [
                {
                    "player_id": "p1",
                    "games_played": 10,
                    "avg": 0.280,
                    "ops": 0.750,
                    "hr": 0.3,
                    "whip": 0.0,
                    "k": 0.0,
                }
            ]
        )
        advanced = pd.DataFrame(
            [
                {
                    "player_id": "p1",
                    "xwoba": float("nan"),
                    "barrel_pct": float("nan"),
                    "xwoba_against": float("nan"),
                    "k_bb_pct": float("nan"),
                }
            ]
        )
        out = apply_shrinkage_to_rates(rates, advanced)
        row = out.iloc[0]
        # No priors → observed preserved
        assert row["ops"] == pytest.approx(0.750)
        assert row["avg"] == pytest.approx(0.280)
        assert not math.isnan(row["ops"])
