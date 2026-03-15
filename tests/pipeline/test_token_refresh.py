"""Tests for src/pipeline/token_refresh.py"""

from __future__ import annotations

from unittest import mock

from src.pipeline.token_refresh import (
    _get_repo_public_key,
    _update_github_secret,
    maybe_write_back_refresh_token,
)


def test_maybe_write_back_no_gh_token(monkeypatch):
    """No-op when GH_TOKEN is not set."""
    monkeypatch.delenv("GH_TOKEN", raising=False)
    result = maybe_write_back_refresh_token("new_token", "old_token")
    assert result is False


def test_maybe_write_back_token_unchanged(monkeypatch):
    """No-op when token has not changed."""
    monkeypatch.setenv("GH_TOKEN", "ghp_test")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    result = maybe_write_back_refresh_token("same_token", "same_token")
    assert result is False


def test_maybe_write_back_no_github_repo(monkeypatch):
    """Logs warning and returns False when GITHUB_REPOSITORY not set."""
    monkeypatch.setenv("GH_TOKEN", "ghp_test")
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    result = maybe_write_back_refresh_token("new_token", "old_token")
    assert result is False


def test_maybe_write_back_pynacl_missing(monkeypatch):
    """Returns False gracefully when PyNaCl is not installed."""
    monkeypatch.setenv("GH_TOKEN", "ghp_test")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

    with mock.patch(
        "src.pipeline.token_refresh._update_github_secret",
        side_effect=ImportError("no pynacl"),
    ):
        result = maybe_write_back_refresh_token("new_token", "old_token")
    assert result is False


def test_maybe_write_back_api_failure(monkeypatch):
    """Returns False (non-fatal) when GitHub API call fails."""
    monkeypatch.setenv("GH_TOKEN", "ghp_test")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

    with mock.patch(
        "src.pipeline.token_refresh._update_github_secret",
        side_effect=Exception("API error"),
    ):
        result = maybe_write_back_refresh_token("new_token", "old_token")
    assert result is False


def test_maybe_write_back_success(monkeypatch):
    """Returns True when token is successfully updated."""
    monkeypatch.setenv("GH_TOKEN", "ghp_test")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

    with mock.patch("src.pipeline.token_refresh._update_github_secret") as mock_update:
        result = maybe_write_back_refresh_token("new_token", "old_token")

    assert result is True
    mock_update.assert_called_once_with(
        "owner", "repo", "YAHOO_REFRESH_TOKEN", "new_token", "ghp_test"
    )


def test_get_repo_public_key_calls_api():
    """_get_repo_public_key makes the correct API call."""
    mock_response = mock.MagicMock()
    mock_response.json.return_value = {"key_id": "123", "key": "base64key"}
    mock_response.raise_for_status = mock.MagicMock()

    with mock.patch(
        "src.pipeline.token_refresh.requests.get", return_value=mock_response
    ) as mock_get:
        result = _get_repo_public_key("owner", "repo", "token")

    assert result == {"key_id": "123", "key": "base64key"}
    mock_get.assert_called_once()
    call_url = mock_get.call_args[0][0]
    assert "owner/repo" in call_url
    assert "public-key" in call_url


def test_update_github_secret_calls_api():
    """_update_github_secret encrypts and PUTs to the API."""
    mock_get_resp = mock.MagicMock()
    mock_get_resp.json.return_value = {
        "key_id": "456",
        "key": "dGVzdA==",
    }  # "test" in base64
    mock_get_resp.raise_for_status = mock.MagicMock()

    mock_put_resp = mock.MagicMock()
    mock_put_resp.raise_for_status = mock.MagicMock()

    with (
        mock.patch(
            "src.pipeline.token_refresh.requests.get", return_value=mock_get_resp
        ),
        mock.patch(
            "src.pipeline.token_refresh.requests.put", return_value=mock_put_resp
        ) as mock_put,
        mock.patch(
            "src.pipeline.token_refresh._encrypt_secret", return_value="encrypted_val"
        ),
    ):
        _update_github_secret("owner", "repo", "MY_SECRET", "secret_val", "token")

    mock_put.assert_called_once()
    put_url = mock_put.call_args[0][0]
    assert "MY_SECRET" in put_url
    put_body = mock_put.call_args[1]["json"]
    assert put_body["encrypted_value"] == "encrypted_val"
    assert put_body["key_id"] == "456"


def test_maybe_write_back_uses_env_var_as_original(monkeypatch):
    """Uses YAHOO_REFRESH_TOKEN env var as original when not provided explicitly."""
    monkeypatch.setenv("GH_TOKEN", "ghp_test")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("YAHOO_REFRESH_TOKEN", "current_env_token")

    # new_token == env var → no writeback
    result = maybe_write_back_refresh_token("current_env_token")
    assert result is False

    # new_token != env var → writeback triggered
    with mock.patch("src.pipeline.token_refresh._update_github_secret"):
        result = maybe_write_back_refresh_token("brand_new_token")
    assert result is True
