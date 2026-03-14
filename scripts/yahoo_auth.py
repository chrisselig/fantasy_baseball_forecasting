#!/usr/bin/env python3
"""
yahoo_auth.py — One-time local Yahoo OAuth 2.0 token generator.

Run this script ONCE locally to generate your access and refresh tokens.
After running, store the four values shown below in GitHub Secrets:

    GitHub Settings → Secrets and variables → Actions → New repository secret

    Secret name               Value
    ─────────────────────────────────────────────────────────────
    YAHOO_CONSUMER_KEY        (your Yahoo app consumer key)
    YAHOO_CONSUMER_SECRET     (your Yahoo app consumer secret)
    YAHOO_ACCESS_TOKEN        (printed below after auth completes)
    YAHOO_REFRESH_TOKEN       (printed below after auth completes)

The GitHub Actions daily pipeline reads these four secrets as environment
variables.  The Shiny app on shinyapps.io does NOT need them — it only
reads from MotherDuck.

Usage:
    # Option 1: export env vars before running
    export YAHOO_CONSUMER_KEY="your_key"
    export YAHOO_CONSUMER_SECRET="your_secret"
    python scripts/yahoo_auth.py

    # Option 2: create a .env file (never commit this file!)
    # .env:
    #   YAHOO_CONSUMER_KEY=your_key
    #   YAHOO_CONSUMER_SECRET=your_secret
    python scripts/yahoo_auth.py

The script will open a browser window for Yahoo login and authorisation.
After approving, it prints the access_token and refresh_token to stdout.

IMPORTANT: Do NOT write the tokens to a file or commit them anywhere.
           Copy them directly to GitHub Secrets.
"""

from __future__ import annotations

import os
import sys


def _load_dotenv() -> None:
    """Load .env file if python-dotenv is available and a .env file exists."""
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]

        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed; rely on shell env


def main() -> None:
    _load_dotenv()

    consumer_key = os.environ.get("YAHOO_CONSUMER_KEY")
    consumer_secret = os.environ.get("YAHOO_CONSUMER_SECRET")

    if not consumer_key or not consumer_secret:
        print(
            "ERROR: YAHOO_CONSUMER_KEY and YAHOO_CONSUMER_SECRET must be set.\n"
            "       Export them as env vars or add them to a .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Starting Yahoo OAuth 2.0 flow...")
    print("A browser window will open for you to log in and authorise the app.")
    print()

    try:
        from yahoo_oauth import OAuth2  # type: ignore[import-untyped]
    except ImportError:
        print(
            "ERROR: yahoo_oauth is not installed.\n       Run: pip install yahoo-oauth",
            file=sys.stderr,
        )
        sys.exit(1)

    # OAuth2 from yahoo_oauth writes a token file by default.
    # We pass a non-persistent in-memory store to avoid writing credentials
    # to disk, then extract the tokens manually.
    oauth = OAuth2(
        consumer_key,
        consumer_secret,
        from_file=None,  # type: ignore[arg-type]
    )

    # The OAuth2 object should now have a valid access token.
    # Depending on the yahoo_oauth version the attributes may differ slightly.
    access_token: str = getattr(oauth, "access_token", None) or getattr(
        oauth, "access_token_key", ""
    )
    refresh_token: str = getattr(oauth, "refresh_token", None) or getattr(
        oauth, "refresh_token_key", ""
    )

    if not access_token:
        print(
            "ERROR: OAuth flow completed but access_token is empty.\n"
            "       Check that the consumer key/secret are correct and that\n"
            "       the Yahoo app has the 'Fantasy Sports' scope enabled.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 60)
    print("OAuth completed successfully!")
    print()
    print("Copy the values below into GitHub Secrets:")
    print()
    print(f"YAHOO_CONSUMER_KEY     = {consumer_key}")
    print(f"YAHOO_CONSUMER_SECRET  = {consumer_secret}")
    print(f"YAHOO_ACCESS_TOKEN     = {access_token}")
    print(f"YAHOO_REFRESH_TOKEN    = {refresh_token}")
    print()
    print("=" * 60)
    print()
    print("REMINDER: Do NOT save these values to any file in this repo.")
    print("          Add them directly to GitHub Secrets.")


if __name__ == "__main__":
    main()
