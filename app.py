# Root-level entrypoint for shinyapps.io deployment.
# shinyapps.io requires app.py at the project root.
# All logic lives in src/app/ — this file just re-exports the app object.

# Load environment from deploy bundle or local .env.
# CI writes _motherduck_env (not in .gitignore, so rsconnect bundles it).
# .env is in .gitignore and gets excluded from the rsconnect bundle.
from pathlib import Path

from dotenv import load_dotenv

_deploy_env = Path(__file__).parent / "_motherduck_env"
if _deploy_env.exists():
    load_dotenv(_deploy_env)
load_dotenv()  # local dev fallback (.env)

from src.app.app import app  # noqa: E402, F401
