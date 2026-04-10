"""Tests for src/app/ modules."""

from __future__ import annotations

import json

import pandas as pd


def test_stub_daily_report_has_required_keys() -> None:
    from src.app.stubs import STUB_DAILY_REPORT

    required_keys = {
        "report_date",
        "week_number",
        "lineup",
        "adds",
        "matchup_summary",
        "ip_pace",
        "callup_alerts",
    }
    assert required_keys.issubset(set(STUB_DAILY_REPORT.keys()))


def test_stub_matchup_summary_has_twelve_categories() -> None:
    from src.app.stubs import STUB_DAILY_REPORT

    summary = STUB_DAILY_REPORT["matchup_summary"]
    assert isinstance(summary, list)
    assert len(summary) == 12


def test_stub_ip_pace_has_required_keys() -> None:
    from src.app.stubs import STUB_DAILY_REPORT

    ip_pace = STUB_DAILY_REPORT["ip_pace"]
    assert isinstance(ip_pace, dict)
    assert {"current_ip", "projected_ip", "min_ip", "on_pace"}.issubset(ip_pace.keys())


def test_stub_roster_df_is_dataframe() -> None:
    from src.app.stubs import STUB_ROSTER_DF

    assert isinstance(STUB_ROSTER_DF, pd.DataFrame)
    assert len(STUB_ROSTER_DF) > 0


def test_stub_waiver_df_is_dataframe() -> None:
    from src.app.stubs import STUB_WAIVER_DF

    assert isinstance(STUB_WAIVER_DF, pd.DataFrame)
    assert len(STUB_WAIVER_DF) > 0


def test_load_daily_report_returns_dict() -> None:
    from src.app.server import _load_daily_report

    report = _load_daily_report()
    assert isinstance(report, dict)


def test_load_roster_returns_dataframe() -> None:
    from src.app.server import _load_roster

    df = _load_roster()
    assert isinstance(df, pd.DataFrame)


def test_waiver_df_from_report_returns_dataframe() -> None:
    from src.app.server import _waiver_df_from_report

    # Empty report → empty DataFrame
    df = _waiver_df_from_report({})
    assert isinstance(df, pd.DataFrame)
    assert df.empty

    # Report with rankings → DataFrame with rank + score columns
    report = {
        "waiver_rankings": [
            {
                "player_id": "422.p.1",
                "player_name": "Test Player",
                "team": "NYY",
                "position": "OF",
                "is_pitcher": False,
                "overall_score": 5.0,
                "fit_score": 2.0,
                "is_callup": False,
                "hr": 1.2,
            }
        ]
    }
    df2 = _waiver_df_from_report(report)
    assert isinstance(df2, pd.DataFrame)
    assert not df2.empty
    assert "rank" in df2.columns
    assert "score" in df2.columns


def test_app_can_be_imported() -> None:
    """Verify the app module imports without error."""
    from src.app.app import app  # noqa: F401

    # Just importing is enough — we don't run the Shiny server in tests


def test_stub_daily_report_is_json_serializable() -> None:
    from src.app.stubs import STUB_DAILY_REPORT

    # Should not raise
    json.dumps(STUB_DAILY_REPORT)


def test_stub_adds_have_required_fields() -> None:
    from src.app.stubs import STUB_DAILY_REPORT

    adds = STUB_DAILY_REPORT["adds"]
    assert isinstance(adds, list)
    for add in adds:
        assert isinstance(add, dict)
        assert "add_player_id" in add
        assert "drop_player_id" in add
        assert "reason" in add
        assert "score" in add
        assert "categories_improved" in add


def test_stub_matchup_summary_categories_include_whip() -> None:
    from src.app.stubs import STUB_DAILY_REPORT

    summary = STUB_DAILY_REPORT["matchup_summary"]
    assert isinstance(summary, list)
    categories = [row["category"] for row in summary]
    assert "whip" in categories


def test_stub_callup_alerts_structure() -> None:
    from src.app.stubs import STUB_DAILY_REPORT

    alerts = STUB_DAILY_REPORT["callup_alerts"]
    assert isinstance(alerts, list)
    for alert in alerts:
        assert isinstance(alert, dict)
        assert "player_id" in alert
        assert "player_name" in alert
        assert "days_since_callup" in alert
        assert "team" in alert
        assert "from_level" in alert


def test_win_pct_class_high() -> None:
    from src.app.server import _win_pct_class

    assert _win_pct_class(0.75) == "win-high"
    assert _win_pct_class(0.65) == "win-high"


def test_win_pct_class_mid() -> None:
    from src.app.server import _win_pct_class

    assert _win_pct_class(0.50) == "win-mid"
    assert _win_pct_class(0.35) == "win-mid"


def test_win_pct_class_low() -> None:
    from src.app.server import _win_pct_class

    assert _win_pct_class(0.10) == "win-low"
    assert _win_pct_class(0.34) == "win-low"


def test_status_class_mapping() -> None:
    from src.app.server import _STATUS_CLASS

    assert _STATUS_CLASS["safe_win"] == "status-safe-win"
    assert _STATUS_CLASS["flippable_win"] == "status-flippable"
    assert _STATUS_CLASS["flippable_loss"] == "status-toss-up"
    assert _STATUS_CLASS["toss_up"] == "status-toss-up"
    assert _STATUS_CLASS["safe_loss"] == "status-safe-loss"


def test_html_table_returns_tag() -> None:
    from htmltools import Tag

    from src.app.server import _html_table

    table = _html_table(["Col A", "Col B"], [["r1c1", "r1c2"], ["r2c1", "r2c2"]])
    assert isinstance(table, Tag)
    html_str = str(table)
    assert "Col A" in html_str
    assert "r1c1" in html_str


def test_html_table_renders_group_headers() -> None:
    from src.app.server import _html_table

    table = _html_table(
        ["A", "B", "C", "D"],
        [["1", "2", "3", "4"]],
        group_headers=[("", 1), ("Weekly", 2), ("Seasonal", 1)],
    )
    html = str(table)
    assert "Weekly" in html
    assert "Seasonal" in html
    assert 'colspan="2"' in html
    assert 'colspan="1"' in html


# ─── _fmt_stat: per-game rate formatting ──────────────────────────────────


def test_fmt_stat_integer_cats_without_per_game_round_to_int() -> None:
    from src.app.server import _fmt_stat

    # Legacy behavior: 0.29 HR rounds to "0".
    assert _fmt_stat(0.29, "hr") == "0"
    assert _fmt_stat(1.6, "sb") == "2"


def test_fmt_stat_per_game_shows_two_decimals_for_integer_cats() -> None:
    from src.app.server import _fmt_stat

    # Waiver/roster callsites use per_game=True so "0.29 HR/game" stays visible.
    assert _fmt_stat(0.29, "hr", per_game=True) == "0.29"
    assert _fmt_stat(0.0, "sb", per_game=True) == "0.00"
    assert _fmt_stat(1.6, "k", per_game=True) == "1.60"


def test_fmt_stat_per_game_does_not_affect_rate_cats() -> None:
    from src.app.server import _fmt_stat

    # WHIP and AVG already use their own precision, per_game is a no-op for them.
    assert _fmt_stat(1.234, "whip", per_game=True) == "1.23"
    assert _fmt_stat(0.287, "avg", per_game=True) == "0.287"


def test_fmt_stat_none_still_returns_dash_when_per_game() -> None:
    from src.app.server import _fmt_stat

    assert _fmt_stat(None, "hr", per_game=True) == "—"
    assert _fmt_stat(float("nan"), "hr", per_game=True) == "—"


# ─── _filter_inactive_waiver_rows ─────────────────────────────────────────


def test_filter_inactive_drops_zero_games_played() -> None:
    from src.app.server import _filter_inactive_waiver_rows

    df = pd.DataFrame(
        [
            {"player_id": "a", "position": "OF,Util", "games_played": 0},
            {"player_id": "b", "position": "OF,Util", "games_played": 7},
        ]
    )
    out = _filter_inactive_waiver_rows(df)
    assert list(out["player_id"]) == ["b"]


def test_filter_inactive_drops_na_and_il_eligibility_tags() -> None:
    from src.app.server import _filter_inactive_waiver_rows

    df = pd.DataFrame(
        [
            {"player_id": "a", "position": "OF,Util,NA", "games_played": 4},
            {"player_id": "b", "position": "3B,Util,IL", "games_played": 4},
            {"player_id": "c", "position": "RP,P,IL10", "games_played": 4},
            {"player_id": "d", "position": "OF,Util", "games_played": 4},
        ]
    )
    out = _filter_inactive_waiver_rows(df)
    assert list(out["player_id"]) == ["d"]


def test_filter_inactive_handles_missing_columns() -> None:
    from src.app.server import _filter_inactive_waiver_rows

    # No games_played / position columns → pass-through.
    df = pd.DataFrame([{"player_id": "a"}, {"player_id": "b"}])
    out = _filter_inactive_waiver_rows(df)
    assert len(out) == 2


def test_filter_inactive_empty_df_returns_empty() -> None:
    from src.app.server import _filter_inactive_waiver_rows

    out = _filter_inactive_waiver_rows(pd.DataFrame())
    assert out.empty


# ─── _tier_for / _color_tier / roster legend ──────────────────────────────


def test_tier_for_hitter_rate_4_tiers() -> None:
    from src.app.server import _tier_for

    # AVG thresholds: elite .300+ / great .270 / avg .240 / poor <.240
    assert _tier_for("avg", 0.310) == "elite"
    assert _tier_for("avg", 0.280) == "great"
    assert _tier_for("avg", 0.250) == "average"
    assert _tier_for("avg", 0.230) == "poor"


def test_tier_for_hitter_counting_3_tier_has_no_elite() -> None:
    from src.app.server import _tier_for

    # HR thresholds: great 2+ / avg 1 / poor 0 — no elite cutoff.
    assert _tier_for("hr", 5) == "great"
    assert _tier_for("hr", 2) == "great"
    assert _tier_for("hr", 1) == "average"
    assert _tier_for("hr", 0) == "poor"


def test_tier_for_pitcher_lower_better() -> None:
    from src.app.server import _tier_for

    # WHIP thresholds: elite <1.00 / great <=1.20 / avg <=1.35 / poor >1.35
    assert _tier_for("whip", 0.95) == "elite"
    assert _tier_for("whip", 1.20) == "great"
    assert _tier_for("whip", 1.30) == "average"
    assert _tier_for("whip", 1.40) == "poor"
    # xwOBA-A: elite <.290
    assert _tier_for("xwoba_against", 0.280) == "elite"
    assert _tier_for("xwoba_against", 0.340) == "poor"


def test_tier_for_launch_angle_range_stat() -> None:
    from src.app.server import _tier_for

    assert _tier_for("avg_launch_angle", 18.0) == "great"  # power zone
    assert _tier_for("avg_launch_angle", 10.0) == "average"  # line drive low
    assert _tier_for("avg_launch_angle", 28.0) == "average"  # line drive high
    assert _tier_for("avg_launch_angle", 4.0) == "poor"  # grounder
    assert _tier_for("avg_launch_angle", 40.0) == "poor"  # pop-up


def test_tier_for_none_and_nan_and_unknown() -> None:
    from src.app.server import _tier_for

    assert _tier_for("hr", None) is None
    assert _tier_for("hr", float("nan")) is None
    assert _tier_for("slot", 0.5) is None  # unknown stat key


def test_color_tier_returns_span_with_color() -> None:
    from htmltools import Tag

    from src.app.server import _TIER_COLORS, _color_tier

    result = _color_tier("0.310", "elite")
    assert isinstance(result, Tag)
    html = str(result)
    assert _TIER_COLORS["elite"] in html
    assert "0.310" in html


def test_color_tier_passes_through_dash_and_none() -> None:
    from src.app.server import _color_tier

    assert _color_tier("—", "elite") == "—"
    assert _color_tier("0.310", None) == "0.310"


def test_fmt_stat_tier_colors_elite_avg() -> None:
    from htmltools import Tag

    from src.app.server import _TIER_COLORS, _fmt_stat_tier

    # .310 AVG → elite, green
    result = _fmt_stat_tier(0.310, "avg")
    assert isinstance(result, Tag)
    assert _TIER_COLORS["elite"] in str(result)


def test_fmt_adv_tier_handles_nan() -> None:
    from src.app.server import _fmt_adv_tier

    # NaN value → plain dash, not a coloured span.
    assert _fmt_adv_tier(float("nan"), "barrel_pct", 1) == "—"


def test_roster_tier_legend_contains_all_four_tiers() -> None:
    from src.app.server import _TIER_COLORS, _roster_tier_legend

    html = str(_roster_tier_legend())
    assert "Elite" in html
    assert "Great" in html
    assert "Average" in html
    assert "Poor" in html
    for color in _TIER_COLORS.values():
        assert color in html
