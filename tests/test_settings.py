"""Unit tests for ``csa_agent.config.Settings`` startup behaviour.

These are plain example tests (no Hypothesis) covering the three
contracts the design pins on configuration loading:

1. Missing or empty ``NEBIUS_API_KEY`` causes a non-zero exit
   (``SystemExit(1)``) with a descriptive message on stderr
   (Requirement 9.4).
2. With a valid environment, :func:`get_settings` returns a
   :class:`Settings` instance (Requirement 9.3).
3. Default field values match the design's "Configuration" table
   (Requirement 6.6).

The conftest fixture ``_reset_csa_caches`` clears the
``functools.lru_cache`` on :func:`get_settings` between tests, so each
case sees a fresh load. We additionally patch
``csa_agent.config.load_dotenv`` to a no-op in every test below so the
real ``.env`` file (which contains a working ``NEBIUS_API_KEY`` for
local development) cannot leak the variable back in after we delete it
with ``monkeypatch.delenv``.
"""

from __future__ import annotations

import os

import pytest

from csa_agent import config as config_module
from csa_agent.config import (
    DEFAULT_CHECKPOINT_DB,
    DEFAULT_DATASET_PATH,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_NEBIUS_BASE_URL,
    DEFAULT_NEBIUS_MODEL,
    DEFAULT_PROFILE_DIR,
    Settings,
    get_settings,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _disable_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop ``get_settings`` from re-reading the repository ``.env``.

    Without this, deleting ``NEBIUS_API_KEY`` from the environment would
    have no observable effect because ``load_dotenv(override=False)``
    inside :func:`get_settings` would re-inject the value sitting in the
    repo's ``.env`` file. Patching it to a no-op keeps the test isolated
    from the developer's local ``.env``.
    """

    monkeypatch.setattr(config_module, "load_dotenv", lambda *a, **kw: False)


def _clear_nebius_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every CSA-related env var so each test has a clean slate."""

    for var in (
        "NEBIUS_API_KEY",
        "NEBIUS_BASE_URL",
        "NEBIUS_MODEL",
        "BITEXT_DATASET_PATH",
        "CHECKPOINT_DB",
        "PROFILE_DIR",
        "MAX_ITERATIONS",
    ):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Missing / empty API key -> SystemExit(1)
# ---------------------------------------------------------------------------


def test_missing_nebius_api_key_triggers_system_exit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Validates Requirement 9.4: unset key -> non-zero exit + message."""

    _disable_dotenv(monkeypatch)
    _clear_nebius_env(monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        get_settings()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "NEBIUS_API_KEY" in captured.err


def test_empty_nebius_api_key_triggers_system_exit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Validates Requirement 9.4: empty/whitespace key -> non-zero exit.

    The loader explicitly trims whitespace and treats empty strings as
    missing, so an env var like ``NEBIUS_API_KEY=   `` must behave the
    same as an unset variable.
    """

    _disable_dotenv(monkeypatch)
    _clear_nebius_env(monkeypatch)
    monkeypatch.setenv("NEBIUS_API_KEY", "   ")

    with pytest.raises(SystemExit) as exc_info:
        get_settings()

    assert exc_info.value.code == 1
    assert "NEBIUS_API_KEY" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Valid env -> Settings instance
# ---------------------------------------------------------------------------


def test_valid_env_yields_settings_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validates Requirement 9.3: a valid key produces a Settings model."""

    _disable_dotenv(monkeypatch)
    _clear_nebius_env(monkeypatch)
    monkeypatch.setenv("NEBIUS_API_KEY", "valid-test-key-123")

    settings = get_settings()

    assert isinstance(settings, Settings)
    assert settings.nebius_api_key == "valid-test-key-123"


def test_valid_env_strips_whitespace_around_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Padding around a real key must be stripped before validation."""

    _disable_dotenv(monkeypatch)
    _clear_nebius_env(monkeypatch)
    monkeypatch.setenv("NEBIUS_API_KEY", "  padded-key  ")

    settings = get_settings()

    assert settings.nebius_api_key == "padded-key"


# ---------------------------------------------------------------------------
# Defaults match the design's Configuration table
# ---------------------------------------------------------------------------


def test_defaults_match_design_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validates Requirement 6.6: every field default matches the design."""

    _disable_dotenv(monkeypatch)
    _clear_nebius_env(monkeypatch)
    # Only the API key is supplied; every other field must take its default.
    monkeypatch.setenv("NEBIUS_API_KEY", "valid-test-key-123")

    settings = get_settings()

    assert settings.nebius_base_url == DEFAULT_NEBIUS_BASE_URL
    assert settings.nebius_base_url == "https://api.studio.nebius.ai/v1/"

    assert settings.nebius_model == DEFAULT_NEBIUS_MODEL
    assert settings.nebius_model == "meta-llama/Llama-3.3-70B-Instruct"

    assert settings.dataset_path == DEFAULT_DATASET_PATH
    assert settings.dataset_path == "./data/bitext_customer_service.csv"

    assert settings.checkpoint_db == DEFAULT_CHECKPOINT_DB
    assert settings.checkpoint_db == "./checkpoints.db"

    assert settings.profile_dir == DEFAULT_PROFILE_DIR
    assert settings.profile_dir == "./profiles"

    assert settings.max_iterations == DEFAULT_MAX_ITERATIONS
    assert settings.max_iterations == 15


def test_settings_is_cached_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Subsequent ``get_settings()`` calls must return the same instance.

    The conftest's autouse fixture clears the cache *between* tests, but
    inside a single test repeated invocations should be free.
    """

    _disable_dotenv(monkeypatch)
    _clear_nebius_env(monkeypatch)
    monkeypatch.setenv("NEBIUS_API_KEY", "valid-test-key-123")

    first = get_settings()
    second = get_settings()

    assert first is second


def test_environment_overrides_take_effect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """An override env var must replace its default value."""

    _disable_dotenv(monkeypatch)
    _clear_nebius_env(monkeypatch)

    custom_dataset = str(tmp_path / "custom.csv")
    monkeypatch.setenv("NEBIUS_API_KEY", "valid-test-key-123")
    monkeypatch.setenv("BITEXT_DATASET_PATH", custom_dataset)
    monkeypatch.setenv("MAX_ITERATIONS", "7")

    settings = get_settings()

    assert settings.dataset_path == custom_dataset
    assert settings.max_iterations == 7
