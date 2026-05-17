"""Checkpointer factory for LangGraph conversation persistence.

This module provides :func:`get_checkpointer`, a context-manager factory that
returns a LangGraph checkpointer suitable for ``graph.compile(checkpointer=...)``.

Backend selection
-----------------

* By default, a :class:`SqliteSaver` is created from
  :func:`SqliteSaver.from_conn_string`, which opens a fresh SQLite database at
  the given path (creating it if necessary).
* When the ``POSTGRES_URL`` environment variable is set to a non-empty value,
  a :class:`PostgresSaver` is created from
  :func:`PostgresSaver.from_conn_string` instead. The Postgres dependency is
  imported lazily so users without ``langgraph-checkpoint-postgres`` installed
  can still run the SQLite path.

API shape
---------

:func:`get_checkpointer` itself is a context manager so callers can write::

    with get_checkpointer(db_path) as saver:
        graph = builder.compile(checkpointer=saver)
        ...

This keeps the underlying connection (SQLite or Postgres) open for the
process lifetime and ensures it is closed cleanly on exit. The shape is
deliberately uniform across backends so callers do not need to special-case
SQLite vs Postgres.

Parent-directory handling
-------------------------

For SQLite, :func:`get_checkpointer` ensures the parent directory of
``db_path`` exists and is writable before opening the connection:

* ``:memory:`` is passed through untouched (no filesystem path).
* If the parent directory is missing, it is created with ``os.makedirs``.
* If creation fails or the resulting directory is not writable, an
  :class:`OSError` is raised with a descriptive message of the form
  ``"Cannot write checkpoint database at <path>: <reason>"`` (Requirement 6.6).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Sentinel SQLite path that means "use a fresh in-memory database".
#: Passed straight through to :func:`SqliteSaver.from_conn_string` without any
#: filesystem checks because there is no parent directory to validate.
_IN_MEMORY_SENTINEL: str = ":memory:"

#: Environment variable that, when set to a non-empty value, switches the
#: backend from SQLite to Postgres.
_POSTGRES_URL_ENV: str = "POSTGRES_URL"


# ---------------------------------------------------------------------------
# Filesystem validation
# ---------------------------------------------------------------------------


def _ensure_parent_writable(db_path: str) -> None:
    """Validate (and if necessary create) the parent directory of ``db_path``.

    The function is a best-effort ``mkdir -p`` followed by a writability
    check. It raises :class:`OSError` with a descriptive message when the
    parent cannot be made into a writable directory.

    Args:
        db_path: Filesystem path where the SQLite database file will live.

    Raises:
        OSError: When the parent directory is missing and cannot be created,
            when the existing parent path is not a directory, or when the
            parent directory is not writable by the current process.
    """

    parent = os.path.dirname(os.path.abspath(db_path))

    # ``os.path.dirname`` of a bare filename like "checkpoints.db" returns the
    # current working directory after ``abspath``, so ``parent`` is always
    # non-empty here. Still, guard defensively.
    if not parent:
        return

    if not os.path.exists(parent):
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as exc:
            raise OSError(
                f"Cannot write checkpoint database at {db_path!r}: "
                f"failed to create parent directory {parent!r} ({exc})."
            ) from exc

    if not os.path.isdir(parent):
        raise OSError(
            f"Cannot write checkpoint database at {db_path!r}: "
            f"parent path {parent!r} exists but is not a directory."
        )

    if not os.access(parent, os.W_OK):
        raise OSError(
            f"Cannot write checkpoint database at {db_path!r}: "
            f"parent directory {parent!r} is not writable."
        )


# ---------------------------------------------------------------------------
# Backend factories
# ---------------------------------------------------------------------------


def _get_postgres_url() -> str | None:
    """Return a non-empty ``POSTGRES_URL`` env var value, or ``None``.

    Whitespace-only strings are normalised to ``None`` so a stray
    ``POSTGRES_URL=`` line behaves like an unset variable.
    """

    raw = os.environ.get(_POSTGRES_URL_ENV)
    if raw is None:
        return None
    trimmed = raw.strip()
    return trimmed or None


@contextmanager
def _sqlite_checkpointer(db_path: str) -> Iterator[Any]:
    """Yield a :class:`SqliteSaver` bound to ``db_path``.

    Imports are local so the SQLite extra is only required when this branch
    actually runs.
    """

    # Local import: keeps top-level import cheap and lets users without
    # ``langgraph-checkpoint-sqlite`` installed still import this module
    # (e.g. in environments that only use Postgres).
    from langgraph.checkpoint.sqlite import SqliteSaver

    if db_path != _IN_MEMORY_SENTINEL:
        _ensure_parent_writable(db_path)

    with SqliteSaver.from_conn_string(db_path) as saver:
        yield saver


@contextmanager
def _postgres_checkpointer(conn_string: str) -> Iterator[Any]:
    """Yield a :class:`PostgresSaver` bound to ``conn_string``.

    The Postgres saver is an optional extra. Importing it lazily means users
    on the SQLite path do not need ``langgraph-checkpoint-postgres``
    installed.
    """

    try:
        # Lazy import so the Postgres extra is only required when actually
        # selected via the ``POSTGRES_URL`` environment variable.
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise ImportError(
            "POSTGRES_URL is set but the 'langgraph-checkpoint-postgres' "
            "package is not installed. Install it (e.g. "
            "`pip install langgraph-checkpoint-postgres`) or unset "
            "POSTGRES_URL to use SQLite."
        ) from exc

    with PostgresSaver.from_conn_string(conn_string) as saver:
        yield saver


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@contextmanager
def get_checkpointer(db_path: str) -> Iterator[Any]:
    """Yield a LangGraph checkpointer bound to ``db_path`` (or Postgres).

    Backend selection follows the rule documented at module level: when the
    ``POSTGRES_URL`` environment variable is set to a non-empty value a
    :class:`PostgresSaver` is yielded; otherwise a :class:`SqliteSaver`
    rooted at ``db_path`` is yielded.

    Use as a context manager so the underlying connection is closed when the
    caller is done::

        with get_checkpointer("./checkpoints.db") as saver:
            graph = builder.compile(checkpointer=saver)
            ...

    Args:
        db_path: Filesystem path for the SQLite database (used only when
            ``POSTGRES_URL`` is not set). Pass ``":memory:"`` for an
            ephemeral in-memory database.

    Yields:
        An entered LangGraph checkpointer ready to be passed to
        ``graph.compile(checkpointer=...)``.

    Raises:
        OSError: When the SQLite path's parent directory is missing and
            cannot be created, exists but is not a directory, or is not
            writable (Requirement 6.6).
        ImportError: When ``POSTGRES_URL`` is set but the
            ``langgraph-checkpoint-postgres`` package is not installed.
    """

    postgres_url = _get_postgres_url()
    if postgres_url is not None:
        with _postgres_checkpointer(postgres_url) as saver:
            yield saver
        return

    with _sqlite_checkpointer(db_path) as saver:
        yield saver


__all__ = ["get_checkpointer"]
