"""API-key resolution, including the guarantee that the key never leaks."""

from __future__ import annotations

import pytest

from cr_labeler.config import ENV_VAR, ApiKey, MissingApiKey, resolve_api_key

SECRET = "AIzaSyTOTALLY-not-a-real-key"


def test_no_key_configured_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setattr("cr_labeler.config._from_keyring", lambda: None)
    assert resolve_api_key(dotenv=tmp_path / "absent.env") is None


def test_environment_variable_is_used(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, SECRET)
    key = resolve_api_key(dotenv=tmp_path / "absent.env")
    assert key is not None
    assert key.value == SECRET
    assert key.source == f"${ENV_VAR}"


def test_key_file_takes_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, "from-environment")
    path = tmp_path / "key.txt"
    path.write_text(SECRET + "\n", encoding="utf-8")
    assert resolve_api_key(key_file=path).value == SECRET


def test_missing_key_file_is_an_error(tmp_path):
    with pytest.raises(MissingApiKey, match="missing or empty"):
        resolve_api_key(key_file=tmp_path / "nope.txt")


def test_dotenv_is_read(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    path = tmp_path / ".env"
    path.write_text(f"# a comment\n{ENV_VAR}='{SECRET}'\nOTHER=x\n", encoding="utf-8")
    key = resolve_api_key(dotenv=path)
    assert key is not None and key.value == SECRET


def test_dotenv_without_the_variable_is_ignored(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setattr("cr_labeler.config._from_keyring", lambda: None)
    path = tmp_path / ".env"
    path.write_text("SOMETHING_ELSE=1\n", encoding="utf-8")
    assert resolve_api_key(dotenv=path) is None


def test_key_never_appears_in_repr_or_str():
    """The whole point of the ApiKey wrapper: it cannot be logged by accident."""
    key = ApiKey(SECRET, "$TEST")
    assert SECRET not in repr(key)
    assert SECRET not in str(key)
    assert SECRET not in f"{key}"
    assert SECRET not in "{}".format(key)  # noqa: UP032 - exercising __format__
    assert "***" in repr(key)


def test_key_is_not_exposed_by_an_exception_traceback():
    key = ApiKey(SECRET, "$TEST")
    try:
        raise RuntimeError(f"failed with {key}")
    except RuntimeError as exc:
        assert SECRET not in str(exc)
