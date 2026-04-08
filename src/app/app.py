"""
app.py

Entry point for the Shiny application.
"""

from __future__ import annotations

import logging
from pathlib import Path

from shiny import App

from src.app.server import server
from src.app.ui import app_ui
from src.db.connection import managed_connection
from src.db.schema import create_all_tables

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting Fantasy Baseball app")

# Ensure all tables exist so queries don't fail on a fresh database.
try:
    with managed_connection() as conn:
        create_all_tables(conn)
except Exception as exc:
    logger.warning("Could not ensure schema on startup: %s", exc)

_WWW = Path(__file__).parent / "www"
app = App(app_ui, server, static_assets=_WWW)
