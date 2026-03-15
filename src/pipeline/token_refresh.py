"""
token_refresh.py

Utility for writing refreshed Yahoo OAuth tokens back to GitHub Secrets.

Yahoo's OAuth 2.0 implementation rotates the refresh token on every access
token refresh. If we don't write the new refresh token back to GitHub Secrets,
the next pipeline run will fail to authenticate.

This module:
1. Compares the current refresh token with what was in the environment at startup
2. If different, encrypts and writes the new token to GitHub Secrets via the API
3. Is a no-op when not running in GitHub Actions (GH_TOKEN not set)

GitHub API requires:
  - GH_TOKEN env var: a token with secrets:write permission
  - GITHUB_REPOSITORY env var: "owner/repo" (auto-set in Actions)
  - PyNaCl for libsodium encryption
"""

from __future__ import annotations

import base64
import logging
import os

import requests

logger = logging.getLogger(__name__)

_GITHUB_API = "https://api.github.com"


def _get_repo_public_key(owner: str, repo: str, gh_token: str) -> dict[str, str]:
    """Fetch the repository's Actions secrets public key.

    Args:
        owner: Repository owner (username or org).
        repo: Repository name.
        gh_token: GitHub token with secrets:read permission.

    Returns:
        Dict with 'key_id' and 'key' (base64-encoded public key).

    Raises:
        requests.HTTPError: If the API request fails.
    """
    resp = requests.get(
        f"{_GITHUB_API}/repos/{owner}/{repo}/actions/secrets/public-key",
        headers={
            "Authorization": f"token {gh_token}",
            "Accept": "application/vnd.github+json",
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()  # type: ignore[no-any-return]


def _encrypt_secret(public_key_b64: str, secret_value: str) -> str:
    """Encrypt a secret value using libsodium (PyNaCl).

    Args:
        public_key_b64: Base64-encoded repository public key.
        secret_value: Plaintext secret to encrypt.

    Returns:
        Base64-encoded encrypted secret.
    """
    from nacl import public  # noqa: PLC0415

    public_key_bytes = base64.b64decode(public_key_b64)
    pk = public.PublicKey(public_key_bytes)
    box = public.SealedBox(pk)
    encrypted = box.encrypt(secret_value.encode("utf-8"))
    return base64.b64encode(encrypted).decode("utf-8")


def _update_github_secret(
    owner: str,
    repo: str,
    secret_name: str,
    secret_value: str,
    gh_token: str,
) -> None:
    """Write a new secret value to GitHub Actions Secrets.

    Args:
        owner: Repository owner.
        repo: Repository name.
        secret_name: Name of the secret to update.
        secret_value: New plaintext value for the secret.
        gh_token: GitHub token with secrets:write permission.

    Raises:
        requests.HTTPError: If the API request fails.
        ImportError: If PyNaCl is not installed.
    """
    key_data = _get_repo_public_key(owner, repo, gh_token)
    encrypted = _encrypt_secret(key_data["key"], secret_value)

    resp = requests.put(
        f"{_GITHUB_API}/repos/{owner}/{repo}/actions/secrets/{secret_name}",
        headers={
            "Authorization": f"token {gh_token}",
            "Accept": "application/vnd.github+json",
        },
        json={"encrypted_value": encrypted, "key_id": key_data["key_id"]},
        timeout=10,
    )
    resp.raise_for_status()
    logger.info("Updated GitHub Secret %s for %s/%s", secret_name, owner, repo)


def maybe_write_back_refresh_token(
    new_refresh_token: str,
    original_refresh_token: str | None = None,
) -> bool:
    """Write back a refreshed Yahoo OAuth token to GitHub Secrets if it changed.

    No-op if:
    - Not running in GitHub Actions (GH_TOKEN not set)
    - Token did not change
    - PyNaCl is not installed

    Args:
        new_refresh_token: The current (possibly refreshed) Yahoo refresh token.
        original_refresh_token: The token value at pipeline start. Defaults to
                                 YAHOO_REFRESH_TOKEN env var.

    Returns:
        True if the secret was updated, False otherwise.
    """
    gh_token = os.environ.get("GH_TOKEN", "")
    if not gh_token:
        logger.debug("GH_TOKEN not set — skipping token writeback (not in Actions)")
        return False

    if original_refresh_token is None:
        original_refresh_token = os.environ.get("YAHOO_REFRESH_TOKEN", "")

    if new_refresh_token == original_refresh_token:
        logger.debug("Refresh token unchanged — no writeback needed")
        return False

    github_repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not github_repo or "/" not in github_repo:
        logger.warning("GITHUB_REPOSITORY not set — cannot write back token")
        return False

    owner, repo = github_repo.split("/", 1)

    try:
        _update_github_secret(
            owner, repo, "YAHOO_REFRESH_TOKEN", new_refresh_token, gh_token
        )
        logger.info("Yahoo refresh token written back to GitHub Secrets")
        return True
    except ImportError:
        logger.warning(
            "PyNaCl not installed — cannot encrypt secret. "
            "Install 'pynacl' to enable token writeback."
        )
        return False
    except Exception as exc:
        logger.warning("Token writeback failed (non-fatal): %s", exc)
        return False
