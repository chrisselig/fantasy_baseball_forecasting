"""
config.py

Typed access to config/league_settings.yaml.

All modules receive a LeagueSettings instance — no module reads the YAML directly.
Category names are normalized to lowercase throughout (matching schema column names).

Usage::

    from src.config import load_league_settings

    settings = load_league_settings()
    settings.category_win_direction   # {'h': 'highest', 'whip': 'lowest', ...}
    settings.active_positions         # ['C', '1B', '2B', ...]  (no BN/IL/NA)
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Path to the config file relative to the project root.
_CONFIG_PATH = Path(__file__).parent.parent / "config" / "league_settings.yaml"

# Roster positions that are NOT active (never started, never scored).
_INACTIVE_SLOTS = frozenset({"BN", "IL", "NA"})


# ── Data models ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CategoryConfig:
    """Configuration for a single scoring category.

    Attributes:
        name: Lowercase category key matching the schema column name
              (e.g. 'h', 'whip', 'k_bb').
        description: Human-readable description from the YAML.
        win_direction: 'highest' or 'lowest'. WHIP is 'lowest'; all others 'highest'.
    """

    name: str
    description: str
    win_direction: str

    def __post_init__(self) -> None:
        if self.win_direction not in ("highest", "lowest"):
            raise ValueError(
                f"Category '{self.name}': win_direction must be 'highest' or 'lowest', "
                f"got '{self.win_direction}'"
            )


@dataclass(frozen=True)
class LeagueSettings:
    """All league settings needed by analysis and pipeline modules.

    Attributes:
        league_id: Yahoo league ID.
        league_name: Display name of the league.
        scoring_type: e.g. 'Head-to-Head - Categories'.
        max_teams: Number of teams in the league.
        batter_categories: Ordered dict of batter scoring categories.
        pitcher_categories: Ordered dict of pitcher scoring categories.
        roster_positions: Full list of roster slots including BN/IL/NA (with duplicates).
        min_ip_per_week: Minimum innings pitched per team per week (21).
        max_acquisitions_per_week: Max adds per team per week (5).
        max_acquisitions_season: Max adds per team per season (None = unlimited).
        trade_end_date: Last date trades are allowed.
        playoff_start_week: First week of playoffs.
        playoff_end_week: Last week of playoffs.
    """

    league_id: int
    league_name: str
    scoring_type: str
    max_teams: int

    batter_categories: dict[str, CategoryConfig]
    pitcher_categories: dict[str, CategoryConfig]

    roster_positions: list[str]

    min_ip_per_week: int
    max_acquisitions_per_week: int
    max_acquisitions_season: int | None

    trade_end_date: datetime.date
    playoff_start_week: int
    playoff_end_week: int
    my_team_key: str = ""  # Yahoo team key e.g. "422.l.87941.t.3"; empty until after draft

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def all_categories(self) -> dict[str, CategoryConfig]:
        """All scoring categories (batters + pitchers) keyed by lowercase name."""
        return {**self.batter_categories, **self.pitcher_categories}

    @property
    def scoring_categories(self) -> list[str]:
        """Ordered list of all category names (lowercase, matching schema columns).

        Order: batter categories first, then pitcher categories.
        """
        return list(self.all_categories.keys())

    @property
    def category_win_direction(self) -> dict[str, str]:
        """Map of category name → 'highest' or 'lowest'.

        Example::

            {'h': 'highest', 'whip': 'lowest', 'k_bb': 'highest', ...}
        """
        return {k: v.win_direction for k, v in self.all_categories.items()}

    @property
    def active_positions(self) -> list[str]:
        """Roster slots that are started and scored (excludes BN, IL, NA).

        Used by the lineup optimizer to know which slots to fill.
        Preserves duplicates (e.g. three 'OF' slots appear three times).
        """
        return [p for p in self.roster_positions if p not in _INACTIVE_SLOTS]

    @property
    def bench_slots(self) -> list[str]:
        """Roster slots that are benched (BN, IL, NA)."""
        return [p for p in self.roster_positions if p in _INACTIVE_SLOTS]


# ── Parser ────────────────────────────────────────────────────────────────────


def _parse_categories(
    raw: dict[str, dict[str, str]],
) -> dict[str, CategoryConfig]:
    """Parse a batter or pitcher category block from the YAML.

    Normalizes category names to lowercase to match schema column names.

    Args:
        raw: Dict from YAML, e.g. {'H': {'desc': 'Hits', 'win': 'highest'}, ...}

    Returns:
        Dict keyed by lowercase category name, values are CategoryConfig instances.

    Raises:
        KeyError: If a category entry is missing 'desc' or 'win'.
    """
    result: dict[str, CategoryConfig] = {}
    for key, attrs in raw.items():
        normalized = key.lower()
        result[normalized] = CategoryConfig(
            name=normalized,
            description=attrs["desc"],
            win_direction=attrs["win"],
        )
    return result


def load_league_settings(config_path: Path | None = None) -> LeagueSettings:
    """Load and validate league settings from the YAML config file.

    Args:
        config_path: Optional override for the config file path.
                     Defaults to config/league_settings.yaml.

    Returns:
        A fully validated LeagueSettings instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If a required field is missing from the YAML.
        ValueError: If a field value is invalid (e.g. bad win_direction).
    """
    path = config_path or _CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"League settings file not found: {path}")

    with path.open() as f:
        raw = yaml.safe_load(f)

    logger.info("Loading league settings from: %s", path)

    batter_cats = _parse_categories(raw["scoring"]["batter_categories"])
    pitcher_cats = _parse_categories(raw["scoring"]["pitcher_categories"])

    trade_end = raw["transactions"]["trade_end_date"]
    if isinstance(trade_end, str):
        trade_end = datetime.date.fromisoformat(trade_end)

    settings = LeagueSettings(
        league_id=int(raw["league"]["id"]),
        league_name=str(raw["league"]["name"]),
        scoring_type=str(raw["league"]["scoring_type"]),
        max_teams=int(raw["league"]["max_teams"]),
        batter_categories=batter_cats,
        pitcher_categories=pitcher_cats,
        roster_positions=list(raw["roster"]["positions"]),
        min_ip_per_week=int(raw["pitching"]["min_innings_pitched_per_week"]),
        max_acquisitions_per_week=int(raw["transactions"]["max_acquisitions_per_week"]),
        max_acquisitions_season=raw["transactions"]["max_acquisitions_season"],
        trade_end_date=trade_end,
        playoff_start_week=int(raw["playoffs"]["start_week"]),
        playoff_end_week=int(raw["playoffs"]["end_week"]),
        my_team_key=str(raw["league"].get("my_team_key", "")),
    )

    logger.info(
        "Loaded settings for league '%s' (ID %d) — %d categories, %d roster slots",
        settings.league_name,
        settings.league_id,
        len(settings.scoring_categories),
        len(settings.roster_positions),
    )
    return settings
