# Root-level entrypoint for shinyapps.io deployment.
# shinyapps.io requires app.py at the project root.
# All logic lives in src/app/ — this file just re-exports the app object.
from src.app.app import app  # noqa: F401
