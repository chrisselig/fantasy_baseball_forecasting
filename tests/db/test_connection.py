"""Tests for src/db/connection.py."""

from __future__ import annotations

import duckdb
import pytest

from src.db.connection import get_connection, is_motherduck, managed_connection


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
