"""Tests for src/config.py."""

from __future__ import annotations

import dataclasses
import datetime
from pathlib import Path

import pytest

from src.config import CategoryConfig, load_league_settings

# ── CategoryConfig ────────────────────────────────────────────────────────────


class TestCategoryConfig:
    def test_valid_highest(self) -> None:
        cat = CategoryConfig(
            name="hr", description="Home Runs", win_direction="highest"
        )
        assert cat.name == "hr"
        assert cat.win_direction == "highest"

    def test_valid_lowest(self) -> None:
        cat = CategoryConfig(name="whip", description="WHIP", win_direction="lowest")
        assert cat.win_direction == "lowest"

    def test_invalid_win_direction_raises(self) -> None:
        with pytest.raises(ValueError, match="win_direction"):
            CategoryConfig(name="era", description="ERA", win_direction="middle")

    def test_frozen(self) -> None:
        cat = CategoryConfig(
            name="k", description="Strikeouts", win_direction="highest"
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            cat.name = "bb"  # type: ignore[misc]


# ── load_league_settings ──────────────────────────────────────────────────────


class TestLoadLeagueSettings:
    def test_loads_without_error(self) -> None:
        settings = load_league_settings()
        assert settings is not None

    def test_league_id(self) -> None:
        settings = load_league_settings()
        assert settings.league_id == 87941

    def test_league_name(self) -> None:
        settings = load_league_settings()
        assert settings.league_name == "The Vlad Guerrero Invitational"

    def test_max_teams(self) -> None:
        settings = load_league_settings()
        assert settings.max_teams == 10

    def test_twelve_scoring_categories(self) -> None:
        settings = load_league_settings()
        assert len(settings.scoring_categories) == 12

    def test_category_names_are_lowercase(self) -> None:
        settings = load_league_settings()
        for name in settings.scoring_categories:
            assert name == name.lower(), f"Category '{name}' is not lowercase"

    def test_batter_categories(self) -> None:
        settings = load_league_settings()
        expected = {"h", "hr", "sb", "bb", "fpct", "avg", "ops"}
        assert set(settings.batter_categories.keys()) == expected

    def test_pitcher_categories(self) -> None:
        settings = load_league_settings()
        expected = {"w", "k", "whip", "k_bb", "sv_h"}
        assert set(settings.pitcher_categories.keys()) == expected

    def test_whip_is_lowest_wins(self) -> None:
        settings = load_league_settings()
        assert settings.category_win_direction["whip"] == "lowest"

    def test_all_others_are_highest_wins(self) -> None:
        settings = load_league_settings()
        for cat, direction in settings.category_win_direction.items():
            if cat != "whip":
                assert direction == "highest", (
                    f"'{cat}' should be 'highest', got '{direction}'"
                )

    def test_roster_has_26_slots(self) -> None:
        settings = load_league_settings()
        assert len(settings.roster_positions) == 26

    def test_active_positions_excludes_bench(self) -> None:
        settings = load_league_settings()
        inactive = {"BN", "IL", "NA"}
        for slot in settings.active_positions:
            assert slot not in inactive, f"'{slot}' should not be in active_positions"

    def test_active_positions_count(self) -> None:
        # 26 total - 4 BN - 4 IL - 2 NA = 16 active
        settings = load_league_settings()
        assert len(settings.active_positions) == 16

    def test_min_ip_per_week(self) -> None:
        settings = load_league_settings()
        assert settings.min_ip_per_week == 21

    def test_max_acquisitions_per_week(self) -> None:
        settings = load_league_settings()
        assert settings.max_acquisitions_per_week == 5

    def test_max_acquisitions_season_is_none(self) -> None:
        settings = load_league_settings()
        assert settings.max_acquisitions_season is None

    def test_trade_end_date(self) -> None:
        settings = load_league_settings()
        assert settings.trade_end_date == datetime.date(2026, 8, 6)

    def test_playoff_weeks(self) -> None:
        settings = load_league_settings()
        assert settings.playoff_start_week == 23
        assert settings.playoff_end_week == 25

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_league_settings(config_path=Path("/nonexistent/path.yaml"))

    def test_category_win_direction_keys_match_scoring_categories(self) -> None:
        settings = load_league_settings()
        assert set(settings.category_win_direction.keys()) == set(
            settings.scoring_categories
        )
