"""
connection.py

MotherDuck connection management.

In production (MOTHERDUCK_TOKEN set): connects to MotherDuck cloud DuckDB.
In local dev / tests (no token): falls back to an in-memory DuckDB instance.
In CI (GITHUB_ACTIONS/CI set) with no token: raises rather than falling back,
so a misconfigured pipeline fails loudly instead of silently writing nothing.

All other modules obtain connections exclusively through this module.
Never construct a duckdb connection outside of get_connection().
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from collections.abc import Callable, Generator

import duckdb

logger = logging.getLogger(__name__)

_DATABASE_NAME = "fantasy_baseball"
_TOKEN_ENV_VAR = "MOTHERDUCK_TOKEN"

# Module-level cached connection for long-lived readers (e.g. the Shiny app),
# guarded by a lock so concurrent reactive renders share a single connection
# instead of each opening a fresh MotherDuck connection.
_shared_connection: duckdb.DuckDBPyConnection | None = None
_shared_lock = threading.Lock()


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection.

    Connects to MotherDuck when MOTHERDUCK_TOKEN is set in the environment.
    Falls back to an in-memory database otherwise (local dev and tests).

    Returns:
        An open DuckDB connection.

    Raises:
        duckdb.Error: If the MotherDuck connection cannot be established.
    """
    token = os.environ.get(_TOKEN_ENV_VAR)
    if token:
        # Pass the token via config rather than embedding it in the connection
        # string so a failed connect cannot echo the credential into logs.
        logger.info("Connecting to MotherDuck database: %s", _DATABASE_NAME)
        conn = duckdb.connect(
            f"md:{_DATABASE_NAME}",
            config={"motherduck_token": token},
        )
        conn.execute(f"USE {_DATABASE_NAME}")
        return conn

    # In CI/GitHub Actions a missing token must fail loudly: otherwise the
    # pipeline silently writes to a throwaway in-memory DB and reports success.
    if os.environ.get("GITHUB_ACTIONS") or os.environ.get("CI"):
        raise RuntimeError(
            f"{_TOKEN_ENV_VAR} is not set but the process is running in CI "
            "(GITHUB_ACTIONS/CI). Refusing to fall back to an in-memory DuckDB, "
            "which would discard all pipeline output. Set the MotherDuck token."
        )

    logger.warning(
        "%s not set — using in-memory DuckDB (data will not persist)",
        _TOKEN_ENV_VAR,
    )
    return duckdb.connect(":memory:")


@contextlib.contextmanager
def managed_connection() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Context manager that opens and safely closes a DuckDB connection.

    Usage::

        with managed_connection() as conn:
            conn.execute("SELECT 1")

    Yields:
        An open DuckDB connection.

    Raises:
        duckdb.Error: If the connection cannot be established.
    """
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def get_shared_connection() -> duckdb.DuckDBPyConnection:
    """Return a process-wide cached DuckDB connection, opening it on first use.

    Intended for the read-only Shiny app, where opening a fresh MotherDuck
    connection in every ``_load_*`` helper adds ~10 network round-trips per
    session start. The connection is created lazily under a lock and reused.

    Returns:
        The shared open DuckDB connection.
    """
    global _shared_connection
    with _shared_lock:
        if _shared_connection is None:
            _shared_connection = get_connection()
        return _shared_connection


def reset_shared_connection() -> None:
    """Close and clear the cached shared connection, if any.

    The next call to :func:`get_shared_connection` or :func:`run_shared`
    will transparently reconnect.
    """
    global _shared_connection
    with _shared_lock:
        conn, _shared_connection = _shared_connection, None
    if conn is not None:
        with contextlib.suppress(Exception):
            conn.close()


def run_shared[T](operation: Callable[[duckdb.DuckDBPyConnection], T]) -> T:
    """Run ``operation`` against the shared connection, reconnecting once on error.

    Serializes access to the cached connection under a lock. If the operation
    raises (e.g. the connection went stale), the cache is dropped, a fresh
    connection is opened, and the operation is retried exactly once.

    Args:
        operation: Callable receiving an open connection and returning a result.

    Returns:
        Whatever ``operation`` returns.

    Raises:
        Exception: Propagated if the operation fails again after reconnecting.
    """
    global _shared_connection
    with _shared_lock:
        if _shared_connection is None:
            _shared_connection = get_connection()
        try:
            return operation(_shared_connection)
        except Exception:
            logger.warning("Shared connection query failed; reconnecting once")
            with contextlib.suppress(Exception):
                _shared_connection.close()
            _shared_connection = get_connection()
            return operation(_shared_connection)


def is_motherduck() -> bool:
    """Return True if the current environment is configured for MotherDuck.

    Useful for conditional logic that should only run in production
    (e.g. skipping schema creation if tables already exist on MotherDuck).

    Returns:
        True if MOTHERDUCK_TOKEN is set, False otherwise.
    """
    return bool(os.environ.get(_TOKEN_ENV_VAR))
