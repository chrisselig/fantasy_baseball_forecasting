"""Tests for data freshness monitoring in server.py."""

import datetime
from unittest import mock

import duckdb

from src.db.schema import FACT_DAILY_REPORTS, create_all_tables


def test_load_data_freshness_offline_when_no_db():
    """Returns is_offline=True when DB is unreachable."""
    from src.app.server import _load_data_freshness

    with mock.patch(
        "src.app.server.managed_connection", side_effect=Exception("no db")
    ):
        result = _load_data_freshness()
    assert result["is_offline"] is True
    assert result["generated_at"] is None


def test_load_data_freshness_fresh_data():
    """Returns is_offline=False with timestamp when report exists."""
    from src.app.server import _load_data_freshness

    conn = duckdb.connect(":memory:")
    create_all_tables(conn)
    today = datetime.date.today()
    generated_at = datetime.datetime.now().isoformat()
    # Insert a fresh report
    conn.execute(
        f"""
        INSERT INTO {FACT_DAILY_REPORTS}
        (report_date, season, week_number, report_json, generated_at, pipeline_run_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        [today, today.year, 1, '{"test": true}', generated_at, "run-1"],
    )

    with mock.patch("src.app.server.managed_connection") as mock_conn_ctx:
        mock_conn_ctx.return_value.__enter__.return_value = conn
        mock_conn_ctx.return_value.__exit__.return_value = False
        result = _load_data_freshness()

    assert result["is_offline"] is False
    assert result["generated_at"] is not None
    conn.close()


def test_load_data_freshness_no_report_today():
    """Returns is_offline=True if no report exists for today."""
    from src.app.server import _load_data_freshness

    conn = duckdb.connect(":memory:")
    create_all_tables(conn)
    # No rows inserted
    with mock.patch("src.app.server.managed_connection") as mock_conn_ctx:
        mock_conn_ctx.return_value.__enter__.return_value = conn
        mock_conn_ctx.return_value.__exit__.return_value = False
        result = _load_data_freshness()
    # No report today — DB is reachable but no data
    assert result["generated_at"] is None
    conn.close()
