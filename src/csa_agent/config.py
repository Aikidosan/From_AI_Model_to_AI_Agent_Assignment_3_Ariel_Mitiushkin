"""Application configuration for the Customer Service Data Analyst Agent.

This module exposes a single :class:`Settings` model whose fields are populated
from environment variables (with optional ``.env`` support via ``python-dotenv``).

The accessor :func:`get_settings` is cached, so configuration is loaded exactly
once per process. If the required ``NEBIUS_API_KEY`` is missing or empty, the
loader writes a descriptive message to ``stderr`` and exits with a non-zero
status code, satisfying Requirement 9.4.

Defaults match the design's "Configuration" table:

================  =========================  ==========================================
Field             Env var                    Default
================  =========================  ==========================================
nebius_api_key    NEBIUS_API_KEY             (required, no default)
nebius_base_url   NEBIUS_BASE_URL            https://api.studio.nebius.ai/v1/
nebius_model      NEBIUS_MODEL               meta-llama/Llama-3.3-70B-Instruct
dataset_path      BITEXT_DATASET_PATH        ./data/bitext_customer_service.csv
checkpoint_db     CHECKPOINT_DB              ./checkpoints.db
profile_dir       PROFILE_DIR                ./profiles
max_iterations    MAX_ITERATIONS             15
================  =========================  ==========================================
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Final

from pydantic import BaseModel, Field

try:  # python-dotenv is an optional-but-listed dependency; degrade gracefully.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised only when dotenv is absent
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        """No-op fallback used when ``python-dotenv`` is not installed."""
        return False


# ---------------------------------------------------------------------------
# Defaults (kept as module-level constants so tests can reference them).
# ---------------------------------------------------------------------------

DEFAULT_NEBIUS_BASE_URL: Final[str] = "https://api.studio.nebius.ai/v1/"
DEFAULT_NEBIUS_MODEL: Final[str] = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_DATASET_PATH: Final[str] = "./data/bitext_customer_service.csv"
DEFAULT_CHECKPOINT_DB: Final[str] = "./checkpoints.db"
DEFAULT_PROFILE_DIR: Final[str] = "./profiles"
DEFAULT_MAX_ITERATIONS: Final[int] = 15


class Settings(BaseModel):
    """Runtime configuration loaded from environment variables.

    All fields are populated by :func:`get_settings`; instantiating ``Settings``
    directly is supported (and useful in tests), but production code should go
    through :func:`get_settings` so caching and the missing-key check apply.
    """

    nebius_api_key: str = Field(
        ...,
        description="API key for the Nebius Token Factory (required).",
    )
    nebius_base_url: str = Field(
        default=DEFAULT_NEBIUS_BASE_URL,
        description="Base URL for the Nebius Token Factory OpenAI-compatible API.",
    )
    nebius_model: str = Field(
        default=DEFAULT_NEBIUS_MODEL,
        description="Default Nebius model identifier used for all LLM calls.",
    )
    dataset_path: str = Field(
        default=DEFAULT_DATASET_PATH,
        description="Filesystem path to the Bitext customer service dataset.",
    )
    checkpoint_db: str = Field(
        default=DEFAULT_CHECKPOINT_DB,
        description="Filesystem path to the SQLite checkpointer database file.",
    )
    profile_dir: str = Field(
        default=DEFAULT_PROFILE_DIR,
        description="Directory where per-user profile JSON files are stored.",
    )
    max_iterations: int = Field(
        default=DEFAULT_MAX_ITERATIONS,
        ge=1,
        description="Maximum ReAct reasoning iterations per query.",
    )


def _read_env(name: str) -> str | None:
    """Return the trimmed value of ``name`` from the environment.

    Whitespace-only or empty strings are normalised to ``None`` so callers can
    treat them as "missing" uniformly.
    """

    raw = os.environ.get(name)
    if raw is None:
        return None
    trimmed = raw.strip()
    return trimmed or None


def _read_int_env(name: str, default: int) -> int:
    """Return an integer env var, falling back to ``default`` when unset/empty.

    Invalid integer strings cause a non-zero exit with a descriptive message,
    matching the strict-startup posture of the Nebius API key check.
    """

    value = _read_env(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        sys.stderr.write(
            f"Invalid integer value for {name}: {value!r}. "
            f"Expected an integer (e.g. {default}).\n"
        )
        sys.exit(1)


def _exit_missing_api_key() -> None:
    """Write a descriptive error to stderr and exit with status 1."""

    sys.stderr.write(
        "NEBIUS_API_KEY is not set. "
        "Set it (e.g. in a .env file or your shell) before launching the agent.\n"
    )
    sys.exit(1)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load, validate, and cache the application :class:`Settings`.

    Behaviour:

    * Loads variables from a ``.env`` file (if present) without overriding
      values already set in the real environment.
    * Trims surrounding whitespace from every value and treats empty strings
      as missing, so a stray ``NEBIUS_API_KEY=  `` line behaves like an unset
      variable.
    * Exits with a non-zero status and a descriptive stderr message when
      ``NEBIUS_API_KEY`` is missing or empty (Requirement 9.4).
    * Subsequent calls return the same cached instance (Requirement 6.6).
    """

    # Load .env without clobbering values that are already set in the
    # process environment (matches python-dotenv's documented default).
    load_dotenv(override=False)

    api_key = _read_env("NEBIUS_API_KEY")
    if api_key is None:
        _exit_missing_api_key()

    return Settings(
        nebius_api_key=api_key,  # type: ignore[arg-type]  # _exit_missing_api_key is NoReturn
        nebius_base_url=_read_env("NEBIUS_BASE_URL") or DEFAULT_NEBIUS_BASE_URL,
        nebius_model=_read_env("NEBIUS_MODEL") or DEFAULT_NEBIUS_MODEL,
        dataset_path=_read_env("BITEXT_DATASET_PATH") or DEFAULT_DATASET_PATH,
        checkpoint_db=_read_env("CHECKPOINT_DB") or DEFAULT_CHECKPOINT_DB,
        profile_dir=_read_env("PROFILE_DIR") or DEFAULT_PROFILE_DIR,
        max_iterations=_read_int_env("MAX_ITERATIONS", DEFAULT_MAX_ITERATIONS),
    )


__all__ = [
    "DEFAULT_CHECKPOINT_DB",
    "DEFAULT_DATASET_PATH",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_NEBIUS_BASE_URL",
    "DEFAULT_NEBIUS_MODEL",
    "DEFAULT_PROFILE_DIR",
    "Settings",
    "get_settings",
]
