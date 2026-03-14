"""
app.py

Entry point for the Shiny application.
"""

from __future__ import annotations

import logging

from shiny import App

from src.app.server import server
from src.app.ui import app_ui

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting Fantasy Baseball app")
app = App(app_ui, server)
