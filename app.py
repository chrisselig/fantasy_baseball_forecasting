# Root-level entrypoint for shinyapps.io deployment.
# shinyapps.io requires app.py at the project root.
# All logic lives in src/app/ — this file just re-exports the app object.

import logging
import os

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("app_startup")

# 1) Try deploy config (written by CI as a Python module — guaranteed import)
try:
    from _deploy_config import MOTHERDUCK_TOKEN  # type: ignore[import-not-found]

    os.environ["MOTHERDUCK_TOKEN"] = MOTHERDUCK_TOKEN
    _logger.info("MotherDuck token loaded from _deploy_config.py")
except ImportError:
    _logger.info("No _deploy_config.py found (expected in local dev)")

# 2) Local dev fallback: load .env
if not os.environ.get("MOTHERDUCK_TOKEN"):
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

# 3) Log connection mode so we can diagnose issues in shinyapps.io logs
if os.environ.get("MOTHERDUCK_TOKEN"):
    _logger.info("MOTHERDUCK_TOKEN is set — will connect to MotherDuck")
else:
    _logger.warning("MOTHERDUCK_TOKEN is NOT set — using in-memory DuckDB (no data!)")

from src.app.app import app  # noqa: E402, F401
