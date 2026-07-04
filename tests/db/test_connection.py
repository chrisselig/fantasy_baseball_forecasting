"""Tests for src/db/connection.py."""

from __future__ import annotations

import duckdb
import pytest

from src.db import connection as connection_module
from src.db.connection import (
    get_connection,
    get_shared_connection,
    is_motherduck,
    managed_connection,
    reset_shared_connection,
    run_shared,
)


@pytest.fixture(autouse=True)
def _clear_ci_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear CI markers so the local in-memory fallback path is exercised.

    Tests that specifically assert the CI guard re-set these vars in their body.
    """
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)


class TestGetConnection:
    def test_returns_in_memory_when_no_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MOTHERDUCK_TOKEN", raising=False)
        conn = get_connection()
        result = conn.execute("SELECT 42 AS n").fetchone()
        assert result is not None
        assert result[0] == 42
        conn.close()

    def test_connection_is_functional(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MOTHERDUCK_TOKEN", raising=False)
        conn = get_connection()
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.execute("INSERT INTO t VALUES (1), (2), (3)")
        total = conn.execute("SELECT SUM(x) FROM t").fetchone()
        assert total is not None
        assert total[0] == 6
        conn.close()

    def test_returns_duckdb_connection_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MOTHERDUCK_TOKEN", raising=False)
        conn = get_connection()
        assert isinstance(conn, duckdb.DuckDBPyConnection)
        conn.close()

    def test_raises_in_github_actions_without_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In GitHub Actions a missing token must fail loudly, not fall back."""
        monkeypatch.delenv("MOTHERDUCK_TOKEN", raising=False)
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        with pytest.raises(RuntimeError, match="MOTHERDUCK_TOKEN"):
            get_connection()

    def test_raises_in_ci_without_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """In generic CI a missing token must fail loudly, not fall back."""
        monkeypatch.delenv("MOTHERDUCK_TOKEN", raising=False)
        monkeypatch.setenv("CI", "true")
        with pytest.raises(RuntimeError, match="MOTHERDUCK_TOKEN"):
            get_connection()


class TestManagedConnection:
    def test_context_manager_yields_connection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MOTHERDUCK_TOKEN", raising=False)
        with managed_connection() as conn:
            assert isinstance(conn, duckdb.DuckDBPyConnection)
            result = conn.execute("SELECT 1").fetchone()
            assert result is not None
            assert result[0] == 1

    def test_connection_closed_after_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MOTHERDUCK_TOKEN", raising=False)
        with managed_connection() as conn:
            pass
        # After context exit, further queries should raise
        with pytest.raises(duckdb.Error):
            conn.execute("SELECT 1")

    def test_connection_closed_on_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MOTHERDUCK_TOKEN", raising=False)
        conn_ref: list[duckdb.DuckDBPyConnection] = []
        with pytest.raises(RuntimeError):
            with managed_connection() as conn:
                conn_ref.append(conn)
                raise RuntimeError("simulated error")
        # Connection should be closed even though an exception was raised
        with pytest.raises(duckdb.Error):
            conn_ref[0].execute("SELECT 1")


class TestIsMotherDuck:
    def test_returns_false_when_no_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MOTHERDUCK_TOKEN", raising=False)
        assert is_motherduck() is False

    def test_returns_true_when_token_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MOTHERDUCK_TOKEN", "fake_token_for_test")
        assert is_motherduck() is True


class TestSharedConnection:
    def teardown_method(self) -> None:
        reset_shared_connection()

    def test_get_shared_connection_is_cached(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MOTHERDUCK_TOKEN", raising=False)
        reset_shared_connection()
        first = get_shared_connection()
        second = get_shared_connection()
        assert first is second

    def test_run_shared_executes_operation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MOTHERDUCK_TOKEN", raising=False)
        reset_shared_connection()
        result = run_shared(lambda conn: conn.execute("SELECT 42").fetchone())
        assert result == (42,)

    def test_run_shared_reconnects_once_on_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MOTHERDUCK_TOKEN", raising=False)
        reset_shared_connection()
        # Prime the cache, then close it out-of-band to simulate a stale conn.
        stale = get_shared_connection()
        stale.close()

        calls: list[int] = []

        def _op(conn: duckdb.DuckDBPyConnection) -> tuple:
            calls.append(1)
            return conn.execute("SELECT 7").fetchone()

        result = run_shared(_op)
        assert result == (7,)
        # First attempt raises on the closed conn, retry succeeds.
        assert len(calls) == 2

    def test_reset_shared_connection_clears_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MOTHERDUCK_TOKEN", raising=False)
        reset_shared_connection()
        first = get_shared_connection()
        reset_shared_connection()
        assert connection_module._shared_connection is None
        second = get_shared_connection()
        assert first is not second
