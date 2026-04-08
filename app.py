# Root-level entrypoint for shinyapps.io deployment.
# shinyapps.io requires app.py at the project root.
# All logic lives in src/app/ — this file just re-exports the app object.

# Load .env file if present (created by CI from GitHub Secrets).
# shinyapps.io does not support rsconnect's -E flag, so the CI deploy
# step writes a .env file into the bundle before uploading.
from dotenv import load_dotenv

load_dotenv()

from src.app.app import app  # noqa: E402, F401
