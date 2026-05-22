"""Example tests for the dataset loader.

These cover the four contracts that pin :func:`csa_agent.dataset.load_dataset`
and :func:`csa_agent.dataset.get_dataset` to the design:

1. Happy-path load: a well-formed CSV produces a DataFrame whose
   columns include every member of :data:`REQUIRED_COLUMNS`
   (Requirements 1.1, 1.3).
2. Missing file: a non-existent path raises ``FileNotFoundError`` with
   a descriptive message (Requirement 1.2).
3. Missing columns: a CSV that lacks a required column raises
   ``ValueError`` listing the missing column(s) (Requirement 1.4).
4. Repeated ``get_dataset()`` calls share a single in-memory frame
   (Requirement 1.5).

The conftest's autouse fixture clears the dataset cache between tests
so each case starts from a clean slate.
"""

from __future__ import annotations

import os
import textwrap

import pandas as pd
import pytest

from csa_agent.dataset import (
    REQUIRED_COLUMNS,
    get_dataset,
    load_dataset,
    reset_dataset_cache,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_csv(path: str, content: str) -> str:
    """Write ``content`` to ``path`` (UTF-8) and return the path.

    ``textwrap.dedent`` is applied so callers can use indented triple
    strings without polluting the CSV with leading whitespace.
    """

    with open(path, "w", encoding="utf-8", newline="") as fp:
        fp.write(textwrap.dedent(content).lstrip("\n"))
    return path


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_load_dataset_happy_path_returns_dataframe(tmp_path) -> None:
    """A well-formed CSV loads into a DataFrame with the required columns.

    Validates Requirements 1.1, 1.3.

    The fixture uses the upstream Bitext column name ``instruction``;
    the loader applies the documented alias ``instruction -> utterance``
    so the resulting frame still satisfies ``REQUIRED_COLUMNS``.
    """

    csv_path = _write_csv(
        str(tmp_path / "good.csv"),
        """\
        instruction,category,intent
        where is my refund,REFUND,track_refund
        i want to cancel my order,ORDER,cancel_order
        please rate my service,FEEDBACK,rate_service
        """,
    )

    df = load_dataset(csv_path)

    assert isinstance(df, pd.DataFrame)
    assert REQUIRED_COLUMNS <= set(df.columns)
    assert len(df) == 3
    assert df.iloc[0]["category"] == "REFUND"
    assert df.iloc[0]["intent"] == "track_refund"
    assert df.iloc[0]["utterance"] == "where is my refund"


def test_load_dataset_accepts_native_utterance_column(tmp_path) -> None:
    """A CSV that already uses ``utterance`` loads without renaming."""

    csv_path = _write_csv(
        str(tmp_path / "native.csv"),
        """\
        utterance,category,intent
        track my order please,ORDER,track_order
        """,
    )

    df = load_dataset(csv_path)

    assert REQUIRED_COLUMNS <= set(df.columns)
    assert df.iloc[0]["utterance"] == "track my order please"


# ---------------------------------------------------------------------------
# Missing file
# ---------------------------------------------------------------------------


def test_load_dataset_missing_file_raises_filenotfounderror(tmp_path) -> None:
    """A non-existent path raises ``FileNotFoundError`` with the path.

    Validates Requirement 1.2.
    """

    missing = str(tmp_path / "does_not_exist.csv")
    assert not os.path.exists(missing)

    with pytest.raises(FileNotFoundError) as exc_info:
        load_dataset(missing)

    # The message should reference the missing path so the user can
    # spot a misconfigured ``BITEXT_DATASET_PATH``.
    assert "does_not_exist.csv" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Missing required column
# ---------------------------------------------------------------------------


def test_load_dataset_missing_column_raises_valueerror_listing_missing(
    tmp_path,
) -> None:
    """A CSV missing a required column raises ``ValueError`` listing it.

    Validates Requirement 1.4. The error must name the offending column
    so the user can fix their dataset without diffing schemas by hand.
    """

    # Drop the ``intent`` column entirely; ``utterance`` and ``category``
    # are present so this is purely a missing-column case.
    csv_path = _write_csv(
        str(tmp_path / "missing_intent.csv"),
        """\
        utterance,category
        where is my refund,REFUND
        """,
    )

    with pytest.raises(ValueError) as exc_info:
        load_dataset(csv_path)

    message = str(exc_info.value)
    assert "intent" in message
    # ``REQUIRED_COLUMNS`` has three members; the message lists the
    # missing ones, not the present ones, so ``utterance`` and
    # ``category`` should *not* show up in the missing-list portion.
    assert "missing required columns" in message.lower()


def test_load_dataset_missing_multiple_columns_lists_each(tmp_path) -> None:
    """When several required columns are absent every name appears."""

    csv_path = _write_csv(
        str(tmp_path / "only_utterance.csv"),
        """\
        utterance
        where is my refund
        """,
    )

    with pytest.raises(ValueError) as exc_info:
        load_dataset(csv_path)

    message = str(exc_info.value)
    assert "category" in message
    assert "intent" in message


# ---------------------------------------------------------------------------
# Cached singleton: get_dataset() returns the same object reference
# ---------------------------------------------------------------------------


def test_get_dataset_caches_singleton(tmp_path) -> None:
    """Repeated ``get_dataset()`` calls return the same DataFrame object.

    Validates Requirement 1.5: tools share a single in-memory frame and
    never reload from disk on each call. We assert object identity
    (``is``) because that is the strongest possible witness of caching.
    """

    csv_path = _write_csv(
        str(tmp_path / "small.csv"),
        """\
        utterance,category,intent
        a,REFUND,track_refund
        b,ORDER,cancel_order
        """,
    )

    # Reset to make this test independent of any prior cached state.
    reset_dataset_cache()

    first = get_dataset(path=csv_path)
    second = get_dataset(path=csv_path)
    third = get_dataset()  # No path arg: should return the cached frame.

    assert first is second
    assert second is third
    assert REQUIRED_COLUMNS <= set(first.columns)


def test_reset_dataset_cache_forces_reload(tmp_path) -> None:
    """After ``reset_dataset_cache``, the next call rebuilds the frame.

    This is the inverse of the singleton test: identity must change
    once the cache is dropped, otherwise tests could not swap fixtures.
    """

    csv_path = _write_csv(
        str(tmp_path / "small.csv"),
        """\
        utterance,category,intent
        a,REFUND,track_refund
        """,
    )

    reset_dataset_cache()
    first = get_dataset(path=csv_path)

    reset_dataset_cache()
    second = get_dataset(path=csv_path)

    assert first is not second
