"""Property tests for the six pure dataset tools.

Feature: customer-service-data-analyst-agent

This module bundles Properties 1-5 covering the pure-function dataset tools
exposed by :func:`csa_agent.tools.core.build_tools`. Each property is
expressed as a Hypothesis test over :func:`tests.strategies.dataframes_st`
and validates a specific requirement from the design document.

* Property 1 -- ``list_categories`` is the deduplicated category column.
  Validates Requirements 3.1.
* Property 2 -- ``filter_by_intent`` / ``filter_by_category`` return exactly
  the matching rows. Validates Requirements 3.2, 3.3.
* Property 3 -- ``count_rows`` is consistent with the filter tools.
  Validates Requirements 3.4.
* Property 4 -- ``get_intent_distribution`` is consistent with
  ``count_rows``. Validates Requirements 3.6.
* Property 5 -- ``show_examples`` is bounded and grounded.
  Validates Requirements 3.5.

Tools are LangChain ``BaseTool`` instances and must be invoked via
``.invoke({...})``; the ``list_categories`` tool takes no arguments and is
called as ``.invoke({})``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from csa_agent.tools.core import FILTER_RESULT_CAP, build_tools

from tests.strategies import (
    category_st,
    dataframes_st,
    intent_st,
    n_examples_st,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tools_by_name(df: pd.DataFrame) -> dict[str, Any]:
    """Return the tool set built from ``df`` keyed by tool name."""

    return {tool.name: tool for tool in build_tools(df)}


def _tool(df: pd.DataFrame, name: str) -> Any:
    """Return the named tool, or fail loudly if it is missing."""

    tools = _tools_by_name(df)
    if name not in tools:
        raise AssertionError(f"tool {name!r} not present in build_tools output")
    return tools[name]


def _is_tool_error(result: Any, expected_value: str) -> bool:
    """Return True iff ``result`` matches the ``ToolError`` envelope shape."""

    return (
        isinstance(result, dict)
        and {"error", "message", "value"} <= set(result.keys())
        and result.get("value") == expected_value
    )


# ---------------------------------------------------------------------------
# Property 1 -- list_categories
# ---------------------------------------------------------------------------

@given(df=dataframes_st())
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_list_categories_is_deduplicated_category_column(df: pd.DataFrame) -> None:
    """Feature: customer-service-data-analyst-agent, Property 1: list_categories is the deduplicated category column.

    Validates Requirements 3.1.

    For any DataFrame ``df``, ``list_categories(df)`` returns a sorted list
    with no duplicates whose set equals ``set(df["category"])`` (after
    dropping NaNs and stringifying).
    """

    list_categories = _tool(df, "list_categories")

    # ``@tool``-decorated callables must be invoked via ``.invoke``; this
    # tool takes no arguments so we pass an empty dict.
    result = list_categories.invoke({})

    expected = {str(value) for value in df["category"].dropna().unique()}

    # Sorted output gives the agent a stable presentation order.
    assert result == sorted(result), (
        f"list_categories result is not sorted: {result!r}"
    )
    # No duplicates.
    assert len(result) == len(set(result)), (
        f"list_categories result contains duplicates: {result!r}"
    )
    # Set equality with the column's distinct values.
    assert set(result) == expected, (
        f"list_categories result {set(result)!r} != expected {expected!r}"
    )


# ---------------------------------------------------------------------------
# Property 2 -- filter_by_category
# ---------------------------------------------------------------------------

@given(df=dataframes_st(), value=category_st())
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_filter_by_category_returns_exactly_matching_rows(
    df: pd.DataFrame, value: str
) -> None:
    """Feature: customer-service-data-analyst-agent, Property 2: filter_by_* returns exactly the matching rows.

    Validates Requirements 3.2, 3.3.
    """

    tool = _tool(df, "filter_by_category")
    result = tool.invoke({"category": value})

    present_values = set(df["category"].astype(str).unique())
    if value not in present_values:
        # Branch: missing value -> structured ToolError envelope.
        assert _is_tool_error(result, value), (
            f"expected ToolError dict for missing category, got {result!r}"
        )
        return

    expected = df[df["category"] == value]
    expected_count = len(expected)

    assert isinstance(result, list)
    assert all(isinstance(row, dict) for row in result)
    # Filter tools cap output at FILTER_RESULT_CAP rows; count_rows is the
    # source of truth for the full count.
    assert len(result) == min(FILTER_RESULT_CAP, expected_count)

    for row in result:
        assert row.get("category") == value

    # With the cap in play we can only require subset containment.
    expected_utterances = set(expected["utterance"].astype(str).tolist())
    returned_utterances = {str(row.get("utterance")) for row in result}
    assert returned_utterances <= expected_utterances


# ---------------------------------------------------------------------------
# Property 2 -- filter_by_intent
# ---------------------------------------------------------------------------

@given(df=dataframes_st(), value=intent_st())
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_filter_by_intent_returns_exactly_matching_rows(
    df: pd.DataFrame, value: str
) -> None:
    """Feature: customer-service-data-analyst-agent, Property 2: filter_by_* returns exactly the matching rows.

    Validates Requirements 3.2, 3.3.
    """

    tool = _tool(df, "filter_by_intent")
    result = tool.invoke({"intent": value})

    present_values = set(df["intent"].astype(str).unique())
    if value not in present_values:
        assert _is_tool_error(result, value), (
            f"expected ToolError dict for missing intent, got {result!r}"
        )
        return

    expected = df[df["intent"] == value]
    expected_count = len(expected)

    assert isinstance(result, list)
    assert all(isinstance(row, dict) for row in result)
    assert len(result) == min(FILTER_RESULT_CAP, expected_count)

    for row in result:
        assert row.get("intent") == value

    expected_utterances = set(expected["utterance"].astype(str).tolist())
    returned_utterances = {str(row.get("utterance")) for row in result}
    assert returned_utterances <= expected_utterances


# ---------------------------------------------------------------------------
# Property 3 -- count_rows
# ---------------------------------------------------------------------------

@given(
    df=dataframes_st(),
    category=st.one_of(st.none(), category_st()),
    intent=st.one_of(st.none(), intent_st()),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_count_rows_is_consistent_with_filter_tools(
    df: pd.DataFrame,
    category: str | None,
    intent: str | None,
) -> None:
    """Feature: customer-service-data-analyst-agent, Property 3: count_rows is consistent with the filter tools.

    Validates Requirements 3.4.

    For any DataFrame ``df`` and any optional ``category`` / ``intent``,
    ``count_rows(df, category, intent)`` equals the size of the row set
    produced by applying the same filters used by ``filter_by_category`` /
    ``filter_by_intent``. Missing values surface as a ``ToolError`` dict.
    """

    tool = _tool(df, "count_rows")
    result = tool.invoke({"category": category, "intent": intent})

    present_categories = set(df["category"].astype(str).unique())
    present_intents = set(df["intent"].astype(str).unique())

    # Branch 1: both filters omitted -> full dataset count.
    if category is None and intent is None:
        assert isinstance(result, int)
        assert result == len(df)
        return

    # Branch 2: category supplied but absent. The implementation checks
    # category first, so this case takes precedence over a missing intent.
    if category is not None and category not in present_categories:
        assert _is_tool_error(result, category), (
            f"expected ToolError dict for missing category, got {result!r}"
        )
        return

    # Branch 3: intent supplied but absent.
    if intent is not None and intent not in present_intents:
        assert _is_tool_error(result, intent), (
            f"expected ToolError dict for missing intent, got {result!r}"
        )
        return

    # Branch 4: every supplied filter is present -> integer count equal to
    # the conjunction of the matching boolean masks. ``True`` collapses
    # cleanly with ``&`` so we can express the conditional masks inline.
    category_mask = (df["category"] == category) if category is not None else True
    intent_mask = (df["intent"] == intent) if intent is not None else True
    expected = int((category_mask & intent_mask).sum())

    assert isinstance(result, int)
    assert result == expected


# ---------------------------------------------------------------------------
# Property 4 -- get_intent_distribution
# ---------------------------------------------------------------------------

@given(df=dataframes_st(), data=st.data())
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_get_intent_distribution_is_consistent_with_count_rows(
    df: pd.DataFrame, data: st.DataObject
) -> None:
    """Feature: customer-service-data-analyst-agent, Property 4: get_intent_distribution is consistent with count_rows.

    Validates Requirements 3.6.

    For any DataFrame ``df`` and any ``category`` present in ``df``,
    ``sum(get_intent_distribution(df, category).values()) ==
    count_rows(df, category=category)``, and every key is an intent that
    appears in rows whose ``category`` equals ``category``.
    """

    # Skip generated frames that have no categories; the property is
    # vacuously true when there is no category to draw.
    categories = sorted(df["category"].dropna().unique().tolist())
    assume(len(categories) > 0)

    # Draw a category guaranteed to be present so the tools take their
    # happy paths. The missing-category branch is covered by Property 6.
    category = data.draw(st.sampled_from(categories))

    tools = _tools_by_name(df)
    distribution = tools["get_intent_distribution"].invoke({"category": category})
    count = tools["count_rows"].invoke({"category": category})

    # Distribution shape: a plain ``dict`` mapping intent strings to ints.
    # A ``ToolError`` would surface as a dict with an ``error`` key, which
    # we explicitly disallow because the category is present.
    assert isinstance(distribution, dict), (
        f"get_intent_distribution returned non-dict: {distribution!r}"
    )
    assert "error" not in distribution, (
        f"get_intent_distribution returned a ToolError for present "
        f"category {category!r}: {distribution!r}"
    )
    for key, value in distribution.items():
        assert isinstance(key, str)
        assert isinstance(value, int)

    # ``count_rows`` must return an int for a present category.
    assert isinstance(count, int), (
        f"count_rows returned non-int for present category {category!r}: "
        f"{count!r}"
    )

    # Sum-consistency: the distribution partitions the rows of the category.
    assert sum(distribution.values()) == count, (
        f"sum(distribution.values())={sum(distribution.values())} does not "
        f"equal count_rows(category={category!r})={count}"
    )

    # Grounding: every key must be an intent that actually appears in
    # rows whose category equals ``category``.
    expected_intents = {
        str(intent)
        for intent in df.loc[df["category"] == category, "intent"].dropna().unique()
    }
    assert set(distribution.keys()) <= expected_intents, (
        f"distribution keys {set(distribution.keys())!r} contain intents "
        f"not present in rows with category={category!r}: {expected_intents!r}"
    )


# ---------------------------------------------------------------------------
# Property 5 -- show_examples
# ---------------------------------------------------------------------------

@given(
    df=dataframes_st(min_rows=1, max_rows=20),
    category=st.one_of(st.none(), category_st()),
    intent=st.one_of(st.none(), intent_st()),
    n=n_examples_st(),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_show_examples_is_bounded_and_grounded(
    df: pd.DataFrame,
    category: str | None,
    intent: str | None,
    n: int,
) -> None:
    """Feature: customer-service-data-analyst-agent, Property 5: show_examples is bounded and grounded.

    Validates Requirements 3.5.

    For any DataFrame ``df``, optional ``category`` / ``intent`` filters,
    and any ``n in [1, 50]``: when both filters resolve, the result is a
    list of strings whose length is at most ``min(n, len(matching_subset))``
    and every returned utterance appears in the matching subset's
    ``utterance`` column. Missing filter values short-circuit to a
    ``ToolError`` dict.
    """

    show_examples = _tool(df, "show_examples")
    result = show_examples.invoke({"category": category, "intent": intent, "n": n})

    category_present = (
        category is None or bool((df["category"] == category).any())
    )
    intent_present = (
        intent is None or bool((df["intent"] == intent).any())
    )

    if category is not None and not category_present:
        assert _is_tool_error(result, category)
        assert result["error"] == "category_not_found"
        return

    if intent is not None and not intent_present:
        assert _is_tool_error(result, intent)
        assert result["error"] == "intent_not_found"
        return

    # Both filters resolve (or are absent): result must be a grounded,
    # bounded list of utterance strings drawn from the matching subset.
    assert isinstance(result, list)
    assert all(isinstance(value, str) for value in result)

    matching_subset = df
    if category is not None:
        matching_subset = matching_subset[matching_subset["category"] == category]
    if intent is not None:
        matching_subset = matching_subset[matching_subset["intent"] == intent]

    matching_count = len(matching_subset)
    assert len(result) <= min(n, matching_count), (
        f"len(result)={len(result)} exceeds min(n={n}, matching_count={matching_count})"
    )

    allowed_utterances = [str(value) for value in matching_subset["utterance"].tolist()]
    for utterance in result:
        assert utterance in allowed_utterances, (
            f"returned utterance {utterance!r} not in matching subset"
        )
