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
    assert "lineup" in report


def test_load_roster_returns_dataframe() -> None:
    from src.app.server import _load_roster

    df = _load_roster()
    assert isinstance(df, pd.DataFrame)


def test_load_waiver_data_returns_dataframe() -> None:
    from src.app.server import _load_waiver_data

    df = _load_waiver_data()
    assert isinstance(df, pd.DataFrame)


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
