"""Dataset loader and module-level cache for the Bitext customer service dataset.

This module is responsible for:

- Loading the dataset from a CSV or Parquet file (chosen by file extension).
- Validating that the schema contains the columns required by every tool.
- Caching the loaded :class:`pandas.DataFrame` as a module-level singleton so
  tools share a single in-memory frame and never reload it per call
  (Requirement 1.5).

Public API:

- :data:`REQUIRED_COLUMNS` -- the columns every dataset must expose.
- :func:`load_dataset` -- pure loader; raises ``FileNotFoundError`` when the
  path is missing (Requirement 1.2) and ``ValueError`` listing the missing
  columns when validation fails (Requirement 1.4).
- :func:`get_dataset` -- cached accessor used by tools and frontends; loads
  the dataset on first call and returns the same frame thereafter
  (Requirement 1.1, 1.5).
- :func:`reset_dataset_cache` -- test hook to drop the cached frame.
"""

from __future__ import annotations

import os
from typing import Final

import pandas as pd

from .config import get_settings


REQUIRED_COLUMNS: Final[set[str]] = {"utterance", "category", "intent"}

# Column-name aliases applied at load time. The Bitext source CSV ships with
# an ``instruction`` column; the design's REQUIRED_COLUMNS uses ``utterance``
# for clarity (it's the customer's spoken/typed text, not a model
# instruction). We rename on read so downstream tools and prompts can use
# the design's vocabulary without the user having to preprocess the CSV.
_COLUMN_ALIASES: Final[dict[str, str]] = {"instruction": "utterance"}

# File extensions handled by :func:`load_dataset`. CSV is the default for the
# Bitext dataset; Parquet is supported for users who pre-convert for speed.
_CSV_EXTENSIONS: Final[frozenset[str]] = frozenset({".csv"})
_PARQUET_EXTENSIONS: Final[frozenset[str]] = frozenset({".parquet", ".pq"})


# Module-level singleton holding the loaded DataFrame.
# Initialised lazily by :func:`get_dataset` and cleared by
# :func:`reset_dataset_cache` (used in tests).
_dataset_cache: pd.DataFrame | None = None


def _read_by_extension(path: str) -> pd.DataFrame:
    """Dispatch to the right pandas reader based on file extension."""

    ext = os.path.splitext(path)[1].lower()
    if ext in _CSV_EXTENSIONS:
        return pd.read_csv(path)
    if ext in _PARQUET_EXTENSIONS:
        return pd.read_parquet(path)
    raise ValueError(
        f"Unsupported dataset file extension {ext!r} for path {path!r}. "
        f"Expected one of: {sorted(_CSV_EXTENSIONS | _PARQUET_EXTENSIONS)}."
    )


def _validate_columns(df: pd.DataFrame, path: str) -> None:
    """Ensure ``df`` exposes every column in :data:`REQUIRED_COLUMNS`.

    Raises ``ValueError`` listing the missing columns and the columns that
    were actually present, satisfying Requirement 1.4.
    """

    present = set(df.columns)
    missing = REQUIRED_COLUMNS - present
    if missing:
        raise ValueError(
            f"Dataset at {path!r} is missing required columns: "
            f"{sorted(missing)}; found: {sorted(present)}."
        )


def load_dataset(path: str) -> pd.DataFrame:
    """Load and validate the dataset at ``path``.

    Args:
        path: Filesystem path to a CSV or Parquet file.

    Returns:
        A :class:`pandas.DataFrame` whose columns include every member of
        :data:`REQUIRED_COLUMNS`.

    Raises:
        FileNotFoundError: When ``path`` does not exist on disk
            (Requirement 1.2).
        ValueError: When the file is loaded but is missing one or more of
            the required columns (Requirement 1.4), or when the file
            extension is not recognised.
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset file not found at {path!r}. "
            f"Set BITEXT_DATASET_PATH or pass an explicit path."
        )

    df = _read_by_extension(path)
    # Apply column-name aliases (e.g. Bitext's "instruction" -> "utterance")
    # before validation so a CSV that only differs in column naming loads
    # cleanly without preprocessing.
    aliases_to_apply = {
        src: dst
        for src, dst in _COLUMN_ALIASES.items()
        if src in df.columns and dst not in df.columns
    }
    if aliases_to_apply:
        df = df.rename(columns=aliases_to_apply)
    _validate_columns(df, path)
    return df


def get_dataset(path: str | None = None) -> pd.DataFrame:
    """Return the loaded dataset, loading it on first call.

    The dataset is cached at module level so subsequent calls return the
    same object reference and tools never reload from disk (Requirement
    1.5).

    Args:
        path: Optional override for the dataset location. When omitted,
            :func:`config.get_settings` provides the default
            (``Settings.dataset_path``).

    Returns:
        The cached :class:`pandas.DataFrame`.
    """

    global _dataset_cache
    if _dataset_cache is None:
        resolved_path = path if path is not None else get_settings().dataset_path
        _dataset_cache = load_dataset(resolved_path)
    return _dataset_cache


def reset_dataset_cache() -> None:
    """Drop the cached dataset so the next :func:`get_dataset` call reloads.

    Intended for use in tests that need to swap fixtures between cases.
    """

    global _dataset_cache
    _dataset_cache = None


__all__ = [
    "REQUIRED_COLUMNS",
    "get_dataset",
    "load_dataset",
    "reset_dataset_cache",
]
