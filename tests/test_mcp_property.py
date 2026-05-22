"""Property test for MCP/direct tool equivalence.

Feature: customer-service-data-analyst-agent
Property 15: MCP tool calls match direct tool calls.

Validates Requirements 8.2, 8.5.

The property has two parts:

1. **Equivalence.** For any valid input, the implementation that the
   FastMCP server invokes returns the same value as a fresh direct call
   to the underlying tool function from
   :func:`csa_agent.tools.core.build_tools`. Because ``mcp_server.py``
   deliberately wires each ``@mcp.tool()`` decorator to the exact
   ``BaseTool.func`` returned by ``build_tools(get_dataset())``, MCP and
   direct calls share one implementation by construction (Requirement
   8.2). This test asserts that property holds across many random valid
   inputs.

2. **Structured error on schema violation.** Constructing a tool input
   that violates the Pydantic schema raises
   :class:`pydantic.ValidationError`, which IS the structured error
   response. FastMCP's automatic Pydantic-driven validation rejects
   invalid inputs before they reach any implementation (Requirement
   8.5), so asserting the schema rejection at the Pydantic level
   captures the same behaviour an MCP client observes.

Tradeoff note (intentional):
Spawning a real FastMCP test client over stdio is brittle on Windows
(subprocess lifecycle, line-buffering, stream framing) and adds
complexity for limited additional coverage given the equivalence-by-
construction design. We test the equivalence at the implementation-
reference level (which is exactly what the FastMCP server invokes) and
the schema enforcement at the Pydantic level (which is exactly what
FastMCP runs before invoking the implementation). This gives us the
two guarantees the spec calls for without inheriting the flakiness of
a process-level roundtrip.
"""

from __future__ import annotations

from typing import Any, Final

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

import mcp_server
from csa_agent.dataset import get_dataset
from csa_agent.tools.core import build_tools
from csa_agent.tools.schemas import (
    CountRowsInput,
    FilterByCategoryInput,
    GetIntentDistributionInput,
    ShowExamplesInput,
)


# ---------------------------------------------------------------------------
# Module-level fixtures: the dataset loads once, and we resolve the direct
# tool implementations once. Both sides (direct and MCP) share the cached
# DataFrame from get_dataset(), so any difference in output between them
# would have to come from the wrapper layer -- which is exactly what
# Property 15 forbids.
# ---------------------------------------------------------------------------

_DF: Final[pd.DataFrame] = get_dataset()
_DIRECT_TOOLS: Final[dict[str, Any]] = {
    tool.name: tool for tool in build_tools(_DF)
}
_DIRECT_LIST_CATEGORIES = _DIRECT_TOOLS["list_categories"].func
_DIRECT_COUNT_ROWS = _DIRECT_TOOLS["count_rows"].func
_DIRECT_SHOW_EXAMPLES = _DIRECT_TOOLS["show_examples"].func
_DIRECT_FILTER_BY_CATEGORY = _DIRECT_TOOLS["filter_by_category"].func
_DIRECT_GET_INTENT_DISTRIBUTION = _DIRECT_TOOLS["get_intent_distribution"].func

# Pre-compute the valid category and intent alphabets so Hypothesis can
# draw inputs that exercise the happy path. The non-existent branch is
# covered separately by Property 6 in test_tools_errors_property.py and
# by the schema-violation branch below.
_CATEGORIES: Final[list[str]] = sorted(_DF["category"].dropna().astype(str).unique().tolist())
_INTENTS: Final[list[str]] = sorted(_DF["intent"].dropna().astype(str).unique().tolist())

assert _CATEGORIES, "dataset must expose at least one category for this test"
assert _INTENTS, "dataset must expose at least one intent for this test"


# ---------------------------------------------------------------------------
# Hypothesis strategies for the valid-input space.
# ---------------------------------------------------------------------------

def _category_st() -> st.SearchStrategy[str]:
    """Sample a category that is guaranteed to exist in the dataset."""
    return st.sampled_from(_CATEGORIES)


def _intent_st() -> st.SearchStrategy[str]:
    """Sample an intent that is guaranteed to exist in the dataset."""
    return st.sampled_from(_INTENTS)


def _optional_category_st() -> st.SearchStrategy[str | None]:
    return st.one_of(st.none(), _category_st())


def _optional_intent_st() -> st.SearchStrategy[str | None]:
    return st.one_of(st.none(), _intent_st())


def _n_st() -> st.SearchStrategy[int]:
    """``show_examples.n`` strategy within the schema-validated 1..50 range."""
    return st.integers(min_value=1, max_value=50)


# ---------------------------------------------------------------------------
# Property 15a -- equivalence on valid inputs
# ---------------------------------------------------------------------------

# Bumped a touch slower than the default deadline because the underlying
# DataFrame has ~27k rows and pandas filtering on every iteration is the
# bottleneck. ``deadline=None`` matches the rest of the suite.
_PBT_SETTINGS = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def test_list_categories_equivalence() -> None:
    """Feature: customer-service-data-analyst-agent, Property 15: MCP tool calls match direct tool calls.

    Validates Requirements 8.2, 8.5.

    ``list_categories`` takes no arguments, so a single call on each
    side is enough to establish equivalence.
    """

    direct = _DIRECT_LIST_CATEGORIES()
    via_mcp = mcp_server._list_categories_impl()

    assert direct == via_mcp, (
        f"list_categories diverged: direct={direct!r}, mcp={via_mcp!r}"
    )


@given(category=_optional_category_st(), intent=_optional_intent_st())
@_PBT_SETTINGS
def test_count_rows_equivalence(
    category: str | None, intent: str | None
) -> None:
    """Feature: customer-service-data-analyst-agent, Property 15: MCP tool calls match direct tool calls.

    Validates Requirements 8.2, 8.5.
    """

    direct = _DIRECT_COUNT_ROWS(category=category, intent=intent)
    via_mcp = mcp_server._count_rows_impl(category=category, intent=intent)

    assert direct == via_mcp, (
        f"count_rows diverged for category={category!r}, intent={intent!r}: "
        f"direct={direct!r}, mcp={via_mcp!r}"
    )


@given(
    category=_optional_category_st(),
    intent=_optional_intent_st(),
    n=_n_st(),
)
@_PBT_SETTINGS
def test_show_examples_equivalence(
    category: str | None, intent: str | None, n: int
) -> None:
    """Feature: customer-service-data-analyst-agent, Property 15: MCP tool calls match direct tool calls.

    Validates Requirements 8.2, 8.5.
    """

    direct = _DIRECT_SHOW_EXAMPLES(category=category, intent=intent, n=n)
    via_mcp = mcp_server._show_examples_impl(
        category=category, intent=intent, n=n
    )

    assert direct == via_mcp, (
        f"show_examples diverged for category={category!r}, intent={intent!r}, "
        f"n={n}: direct={direct!r}, mcp={via_mcp!r}"
    )


@given(category=_category_st())
@_PBT_SETTINGS
def test_filter_by_category_equivalence(category: str) -> None:
    """Feature: customer-service-data-analyst-agent, Property 15: MCP tool calls match direct tool calls.

    Validates Requirements 8.2, 8.5.
    """

    direct = _DIRECT_FILTER_BY_CATEGORY(category=category)
    via_mcp = mcp_server._filter_by_category_impl(category=category)

    assert direct == via_mcp, (
        f"filter_by_category diverged for category={category!r}: "
        f"len(direct)={len(direct) if isinstance(direct, list) else 'n/a'}, "
        f"len(mcp)={len(via_mcp) if isinstance(via_mcp, list) else 'n/a'}"
    )


@given(category=_category_st())
@_PBT_SETTINGS
def test_get_intent_distribution_equivalence(category: str) -> None:
    """Feature: customer-service-data-analyst-agent, Property 15: MCP tool calls match direct tool calls.

    Validates Requirements 8.2, 8.5.
    """

    direct = _DIRECT_GET_INTENT_DISTRIBUTION(category=category)
    via_mcp = mcp_server._get_intent_distribution_impl(category=category)

    assert direct == via_mcp, (
        f"get_intent_distribution diverged for category={category!r}: "
        f"direct={direct!r}, mcp={via_mcp!r}"
    )


# ---------------------------------------------------------------------------
# Property 15b -- structured error on schema violation
#
# These assertions stand in for the FastMCP roundtrip: FastMCP infers
# JSON schemas from the same Pydantic models we test here, validates
# inbound arguments against them, and surfaces a structured error to the
# MCP client when validation fails. Asserting the Pydantic raise at the
# model level is equivalent and avoids the subprocess/transport
# brittleness called out in the module docstring.
# ---------------------------------------------------------------------------


@given(n=st.integers(max_value=0))
@_PBT_SETTINGS
def test_show_examples_schema_rejects_n_below_minimum(n: int) -> None:
    """Feature: customer-service-data-analyst-agent, Property 15: MCP tool calls match direct tool calls.

    Validates Requirements 8.5.

    ``ShowExamplesInput.n`` is constrained to ``[1, 50]``. Any value
    below 1 must surface as a Pydantic ``ValidationError`` -- the
    structured error response the MCP client receives -- and never reach
    the implementation.
    """

    with pytest.raises(ValidationError):
        ShowExamplesInput(n=n)


@given(n=st.integers(min_value=51, max_value=10_000))
@_PBT_SETTINGS
def test_show_examples_schema_rejects_n_above_maximum(n: int) -> None:
    """Feature: customer-service-data-analyst-agent, Property 15: MCP tool calls match direct tool calls.

    Validates Requirements 8.5.

    Symmetric upper-bound check on ``ShowExamplesInput.n``. The example
    in the spec uses ``n=99``; we generalise to all values above 50.
    """

    with pytest.raises(ValidationError):
        ShowExamplesInput(n=n)


def test_show_examples_schema_rejects_n_99_explicitly() -> None:
    """Feature: customer-service-data-analyst-agent, Property 15: MCP tool calls match direct tool calls.

    Validates Requirements 8.5.

    Explicit pinned example matching the spec excerpt: ``n=99`` must
    raise :class:`pydantic.ValidationError`.
    """

    with pytest.raises(ValidationError):
        ShowExamplesInput(n=99)


def test_filter_by_category_schema_requires_category() -> None:
    """Feature: customer-service-data-analyst-agent, Property 15: MCP tool calls match direct tool calls.

    Validates Requirements 8.5.

    ``FilterByCategoryInput.category`` is required; omitting it must
    surface as a structured ``ValidationError``.
    """

    with pytest.raises(ValidationError):
        FilterByCategoryInput()  # type: ignore[call-arg]


def test_get_intent_distribution_schema_requires_category() -> None:
    """Feature: customer-service-data-analyst-agent, Property 15: MCP tool calls match direct tool calls.

    Validates Requirements 8.5.

    ``GetIntentDistributionInput.category`` is required; omitting it
    must surface as a structured ``ValidationError``.
    """

    with pytest.raises(ValidationError):
        GetIntentDistributionInput()  # type: ignore[call-arg]


def test_count_rows_schema_rejects_wrong_types() -> None:
    """Feature: customer-service-data-analyst-agent, Property 15: MCP tool calls match direct tool calls.

    Validates Requirements 8.5.

    ``CountRowsInput`` accepts only ``str | None`` for both filters.
    Passing a non-string (e.g. an integer) must surface as a structured
    ``ValidationError`` rather than reaching the implementation.
    """

    with pytest.raises(ValidationError):
        CountRowsInput(category=123)  # type: ignore[arg-type]
