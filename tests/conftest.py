"""Shared pytest configuration for the Customer Service Data Analyst Agent tests.

Responsibilities:

* Insert ``src/`` onto ``sys.path`` so test modules can import ``csa_agent.*``
  without an editable install.
* Provide a stub ``NEBIUS_API_KEY`` so :func:`csa_agent.config.get_settings`
  does not exit non-zero during test collection. Tests that need to
  exercise the missing-key behaviour clear it locally via ``monkeypatch``.
* Reset the cached :func:`get_settings` and :func:`get_dataset` singletons
  between tests so per-test environment overrides take effect cleanly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Stub key for offline tests; tests that need to assert the missing-key
# behaviour (e.g. test_settings.py) will clear this via monkeypatch.
os.environ.setdefault("NEBIUS_API_KEY", "stub-test-key")


import pytest  # noqa: E402  -- after sys.path setup


@pytest.fixture(autouse=True)
def _reset_csa_caches():
    """Reset cached singletons between tests so env overrides take effect."""
    # Reset get_settings cache.
    try:
        from csa_agent.config import get_settings as _get_settings
        _get_settings.cache_clear()
    except Exception:
        pass
    # Reset dataset singleton.
    try:
        from csa_agent.dataset import reset_dataset_cache
        reset_dataset_cache()
    except Exception:
        pass
    # Reset summarize sub-agent cache (built lazily inside nodes.py).
    try:
        from csa_agent.nodes import reset_summarize_subagent_cache
        reset_summarize_subagent_cache()
    except Exception:
        pass
    yield
