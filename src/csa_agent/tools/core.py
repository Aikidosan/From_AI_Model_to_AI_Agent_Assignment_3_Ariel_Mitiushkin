"""Core dataset tools wrapped as LangChain ``BaseTool`` instances.

This module implements the seven dataset tools required by the agent:

- :func:`list_categories` -- distinct category values (Requirement 3.1).
- :func:`filter_by_intent` -- rows matching an intent (Requirement 3.2).
- :func:`filter_by_category` -- rows matching a category (Requirement 3.3).
- :func:`count_rows` -- count with optional category/intent filters (Requirement 3.4).
- :func:`show_examples` -- up to ``n`` representative utterances (Requirement 3.5).
- :func:`get_intent_distribution` -- intent -> count for a category (Requirement 3.6).
- :func:`summarize_category` -- LLM-generated narrative grounded in dataset
  utterances (Requirements 3.7, 3.8, 9.1).

Design contract:

* All tools are pure functions over a captured :class:`pandas.DataFrame`. They
  never reload the dataset and never mutate it.
* ``filter_by_intent`` and ``filter_by_category`` cap their result at
  :data:`FILTER_RESULT_CAP` rows (100) to keep the agent's token budget
  bounded; the full count is always available via ``count_rows``.
* ``show_examples`` defensively clamps ``n`` to ``[1, 50]`` even though the
  Pydantic schema validates the same range (Requirement 3.5). This protects
  the function when callers bypass schema validation (e.g. the FastMCP
  endpoint upstream of the LangChain wrapper).
* On an unknown category or intent, every tool returns a :class:`ToolError`
  shaped ``dict`` (``{"error": ..., "message": ..., "value": ...}``) and
  never raises (Requirement 3.8). This lets the ReAct agent observe a
  recoverable result and self-correct (e.g. by calling ``list_categories``
  and retrying).
* Each tool is wrapped with LangChain's ``@tool`` decorator and bound to the
  matching Pydantic input schema (Requirement 3.9). ``build_tools`` returns
  them as a ``list[BaseTool]`` for direct consumption by
  ``create_react_agent``.

Usage::

    from csa_agent.dataset import get_dataset
    from csa_agent.tools.core import build_tools

    tools = build_tools(get_dataset())
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool

from ..llm import get_llm
from .schemas import (
    CountRowsInput,
    FilterByCategoryInput,
    FilterByIntentInput,
    GetIntentDistributionInput,
    ListCategoriesInput,
    ShowExamplesInput,
    SummarizeCategoryInput,
    ToolError,
)


# Maximum number of rows returned by filter tools. ``count_rows`` always
# reports the *full* count irrespective of this cap (Requirement 3.4).
FILTER_RESULT_CAP: int = 100

# Defensive bounds for ``show_examples.n``. The Pydantic schema enforces the
# same range, but the tool clamps independently so the function is safe even
# when invoked outside the schema-validated path.
_MIN_EXAMPLES: int = 1
_MAX_EXAMPLES: int = 50

# Number of representative utterances sampled when summarizing a category.
# Kept small enough to fit comfortably in the prompt while still giving the
# model a faithful cross-section of the category to ground its summary in.
_SUMMARIZE_SAMPLE_SIZE: int = 10

# System prompt that frames the summarization task and forbids speculation
# beyond the supplied utterances (Requirement 3.7).
_SUMMARIZE_SYSTEM_PROMPT: str = (
    "You are a data analyst summarizing utterances from a customer service "
    "dataset. Ground your summary in the provided examples; do not invent "
    "details, statistics, or behaviors that are not supported by the "
    "examples. Keep the summary concise (2-3 sentences)."
)


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

def _category_not_found(value: str) -> dict[str, Any]:
    """Return a :class:`ToolError`-shaped dict for an unknown category."""

    return ToolError(
        error="category_not_found",
        message=(
            f"Category {value!r} was not found in the dataset. "
            f"Call list_categories to see the valid options."
        ),
        value=value,
    ).model_dump()


def _intent_not_found(value: str) -> dict[str, Any]:
    """Return a :class:`ToolError`-shaped dict for an unknown intent."""

    return ToolError(
        error="intent_not_found",
        message=(
            f"Intent {value!r} was not found in the dataset. "
            f"Use filter_by_category or get_intent_distribution to discover "
            f"valid intents."
        ),
        value=value,
    ).model_dump()


def _category_exists(df: pd.DataFrame, category: str) -> bool:
    """Return True iff ``category`` appears in the ``category`` column."""

    return bool((df["category"] == category).any())


def _intent_exists(df: pd.DataFrame, intent: str) -> bool:
    """Return True iff ``intent`` appears in the ``intent`` column."""

    return bool((df["intent"] == intent).any())


def _records(subset: pd.DataFrame, cap: int) -> list[dict[str, Any]]:
    """Convert at most ``cap`` rows of ``subset`` to a list of plain dicts."""

    return subset.head(cap).to_dict(orient="records")


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

def build_tools(df: pd.DataFrame) -> list[BaseTool]:
    """Build the dataset tools bound to ``df`` and return them as ``BaseTool``s.

    The returned list contains all seven tools required by Requirement
    3.1-3.9: six pure-function tools over the captured ``df`` plus the
    LLM-backed :func:`summarize_category` (Requirement 3.7), which calls
    :func:`csa_agent.llm.get_llm` lazily on each invocation so test doubles
    can patch the factory.

    Args:
        df: The Bitext customer service DataFrame loaded by
            :func:`csa_agent.dataset.get_dataset`. The frame must expose at
            least the ``utterance``, ``category`` and ``intent`` columns
            validated by :func:`csa_agent.dataset.load_dataset`.

    Returns:
        A list of LangChain :class:`BaseTool` instances ready to be passed to
        ``create_react_agent`` or registered on the FastMCP server.
    """

    @tool("list_categories", args_schema=ListCategoriesInput)
    def list_categories() -> list[str]:
        """List the distinct category values present in the dataset.

        Returns:
            Sorted list of unique category names. Use this before calling
            ``filter_by_category`` or ``get_intent_distribution`` when the
            user's wording does not match a known category exactly.
        """

        # ``dropna`` guards against stray NaN values; ``sorted`` gives the
        # agent a stable presentation order.
        unique = df["category"].dropna().unique().tolist()
        return sorted(str(value) for value in unique)

    @tool("filter_by_intent", args_schema=FilterByIntentInput)
    def filter_by_intent(intent: str) -> list[dict[str, Any]] | dict[str, Any]:
        """Return rows whose ``intent`` column equals ``intent``.

        At most :data:`FILTER_RESULT_CAP` rows are returned to keep token
        usage bounded; use ``count_rows`` for the full total. On an unknown
        intent, returns a structured :class:`ToolError`-shaped dict instead
        of raising.

        Args:
            intent: Exact intent name (e.g. ``"track_refund"``).
        """

        if not _intent_exists(df, intent):
            return _intent_not_found(intent)
        subset = df[df["intent"] == intent]
        return _records(subset, FILTER_RESULT_CAP)

    @tool("filter_by_category", args_schema=FilterByCategoryInput)
    def filter_by_category(category: str) -> list[dict[str, Any]] | dict[str, Any]:
        """Return rows whose ``category`` column equals ``category``.

        At most :data:`FILTER_RESULT_CAP` rows are returned; use
        ``count_rows`` for the full total. On an unknown category, returns a
        structured :class:`ToolError`-shaped dict instead of raising.

        Args:
            category: Exact category name (e.g. ``"REFUND"``).
        """

        if not _category_exists(df, category):
            return _category_not_found(category)
        subset = df[df["category"] == category]
        return _records(subset, FILTER_RESULT_CAP)

    @tool("count_rows", args_schema=CountRowsInput)
    def count_rows(
        category: str | None = None,
        intent: str | None = None,
    ) -> int | dict[str, Any]:
        """Count rows matching the optional category and/or intent filters.

        Both filters are optional. When both are omitted, returns the size of
        the full dataset. When either filter references a value not present
        in the dataset, returns a :class:`ToolError`-shaped dict.

        Args:
            category: Optional exact category name.
            intent: Optional exact intent name.

        Returns:
            The integer count of matching rows, or a ``ToolError`` dict.
        """

        if category is not None and not _category_exists(df, category):
            return _category_not_found(category)
        if intent is not None and not _intent_exists(df, intent):
            return _intent_not_found(intent)

        subset = df
        if category is not None:
            subset = subset[subset["category"] == category]
        if intent is not None:
            subset = subset[subset["intent"] == intent]
        return int(len(subset))

    @tool("show_examples", args_schema=ShowExamplesInput)
    def show_examples(
        category: str | None = None,
        intent: str | None = None,
        n: int = 5,
    ) -> list[str] | dict[str, Any]:
        """Return up to ``n`` representative utterances for a category/intent.

        ``n`` is defensively clamped to ``[1, 50]`` even though the Pydantic
        schema enforces the same range (Requirement 3.5). When either filter
        references a value not present in the dataset, returns a
        :class:`ToolError`-shaped dict.

        Args:
            category: Optional exact category name.
            intent: Optional exact intent name.
            n: Number of examples to return (clamped to ``[1, 50]``).

        Returns:
            Up to ``min(n, matching_count)`` utterance strings, or a
            ``ToolError`` dict.
        """

        n_clamped = max(_MIN_EXAMPLES, min(_MAX_EXAMPLES, int(n)))

        if category is not None and not _category_exists(df, category):
            return _category_not_found(category)
        if intent is not None and not _intent_exists(df, intent):
            return _intent_not_found(intent)

        subset = df
        if category is not None:
            subset = subset[subset["category"] == category]
        if intent is not None:
            subset = subset[subset["intent"] == intent]
        return [str(value) for value in subset["utterance"].head(n_clamped).tolist()]

    @tool("get_intent_distribution", args_schema=GetIntentDistributionInput)
    def get_intent_distribution(category: str) -> dict[str, int] | dict[str, Any]:
        """Return ``intent -> count`` for every intent within ``category``.

        On an unknown category, returns a :class:`ToolError`-shaped dict
        instead of raising.

        Args:
            category: Exact category name.

        Returns:
            A mapping from intent name to its row count within the category,
            or a ``ToolError`` dict. The sum of values equals
            ``count_rows(category=category)``.
        """

        if not _category_exists(df, category):
            return _category_not_found(category)
        subset = df[df["category"] == category]
        counts = subset["intent"].value_counts(dropna=True)
        return {str(intent): int(count) for intent, count in counts.items()}

    @tool("summarize_category", args_schema=SummarizeCategoryInput)
    def summarize_category(category: str) -> str | dict[str, Any]:
        """Summarize the utterances in ``category`` using a Nebius LLM call.

        Samples up to :data:`_SUMMARIZE_SAMPLE_SIZE` representative utterances
        from the category (deterministic ``head`` order so repeated calls are
        reproducible) and asks the LLM for a short natural-language summary
        grounded strictly in those examples (Requirement 3.7). All LLM calls
        flow through :func:`csa_agent.llm.get_llm` (Requirement 9.1).

        On an unknown category, returns a :class:`ToolError`-shaped dict
        instead of raising (Requirement 3.8).

        Args:
            category: Exact category name.

        Returns:
            A short (2-3 sentence) summary string, or a ``ToolError`` dict.
        """

        if not _category_exists(df, category):
            return _category_not_found(category)

        subset = df[df["category"] == category]
        sample_size = min(_SUMMARIZE_SAMPLE_SIZE, len(subset))
        # ``head`` is deterministic and the Bitext dataset is naturally
        # ordered, so this gives a representative cross-section without the
        # noise of a random seed in production logs.
        utterances = [
            str(value) for value in subset["utterance"].head(sample_size).tolist()
        ]

        # Render the utterances verbatim, numbered, so the model can refer
        # to them precisely and so a human reading the prompt can verify
        # grounding.
        rendered_examples = "\n".join(
            f"{idx}. {utterance}" for idx, utterance in enumerate(utterances, start=1)
        )
        human_prompt = (
            f"Category: {category}\n\n"
            f"Representative utterances ({sample_size} of {len(subset)} total "
            f"rows):\n{rendered_examples}\n\n"
            "Write a 2-3 sentence summary describing the kinds of customer "
            "needs and language captured by this category. Stay grounded in "
            "the utterances above; do not invent details."
        )

        llm = get_llm()
        response = llm.invoke(
            [
                SystemMessage(content=_SUMMARIZE_SYSTEM_PROMPT),
                HumanMessage(content=human_prompt),
            ]
        )
        return str(response.content)

    return [
        list_categories,
        filter_by_intent,
        filter_by_category,
        count_rows,
        show_examples,
        get_intent_distribution,
        summarize_category,
    ]


__all__ = ["FILTER_RESULT_CAP", "build_tools"]
