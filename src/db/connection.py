"""
connection.py

MotherDuck connection management.

In production (MOTHERDUCK_TOKEN set): connects to MotherDuck cloud DuckDB.
In local dev / tests (no token): falls back to an in-memory DuckDB instance.

All other modules obtain connections exclusively through this module.
Never construct a duckdb connection outside of get_connection().
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Generator

import duckdb

logger = logging.getLogger(__name__)

_DATABASE_NAME = "fantasy_baseball"
_TOKEN_ENV_VAR = "MOTHERDUCK_TOKEN"


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
        connection_string = f"md:{_DATABASE_NAME}?motherduck_token={token}"
        logger.info("Connecting to MotherDuck database: %s", _DATABASE_NAME)
        return duckdb.connect(connection_string)

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


def is_motherduck() -> bool:
    """Return True if the current environment is configured for MotherDuck.

    Useful for conditional logic that should only run in production
    (e.g. skipping schema creation if tables already exist on MotherDuck).

    Returns:
        True if MOTHERDUCK_TOKEN is set, False otherwise.
    """
    return bool(os.environ.get(_TOKEN_ENV_VAR))
