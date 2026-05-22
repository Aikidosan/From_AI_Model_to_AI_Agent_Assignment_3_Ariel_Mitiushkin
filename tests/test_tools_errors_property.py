"""Property test for dataset tool error handling.

Feature: customer-service-data-analyst-agent
Property 6: Tools return structured errors for missing values.

Validates Requirements 3.8.

For any string ``s`` that does not appear as a category or intent in
``df``, every dataset tool that takes a category/intent parameter returns
a :class:`csa_agent.tools.schemas.ToolError`-shaped dict (with ``error``,
``message`` and ``value`` fields) and does not raise.

Tools exercised:

* Required ``category`` / ``intent`` parameter:
  ``filter_by_intent``, ``filter_by_category``,
  ``get_intent_distribution``, ``summarize_category``.
* Optional filters that may carry a bogus value:
  ``count_rows`` and ``show_examples`` (tested with the bogus value
  passed as both ``category`` and ``intent``).

``summarize_category`` calls :func:`csa_agent.llm.get_llm` internally.
The function checks the category exists *first* and only calls the LLM
when the category is valid, so a bogus category never triggers a real
network call. Even so, we monkeypatch ``get_llm`` to return a
:class:`tests.fakes.FakeChatModel` as a safety net so this test can
never escape into a live Nebius request.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from csa_agent.tools.core import build_tools

from tests.fakes import FakeChatModel
from tests.strategies import dataframes_st, non_existent_string_st


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOOL_ERROR_KEYS = {"error", "message", "value"}
_VALID_ERROR_CODES = {"category_not_found", "intent_not_found"}


def _tools_by_name(df: pd.DataFrame) -> dict[str, Any]:
    """Return the freshly built tool set keyed by tool name."""

    return {tool.name: tool for tool in build_tools(df)}


def _assert_tool_error(result: Any, expected_value: str) -> None:
    """Assert ``result`` matches the ``ToolError`` envelope contract."""

    assert isinstance(result, dict), (
        f"expected ToolError dict, got {type(result).__name__}: {result!r}"
    )
    assert _TOOL_ERROR_KEYS <= set(result.keys()), (
        f"expected keys {_TOOL_ERROR_KEYS}, got {set(result.keys())}"
    )
    assert result["value"] == expected_value, (
        f"expected value={expected_value!r}, got {result['value']!r}"
    )
    assert result["error"] in _VALID_ERROR_CODES, (
        f"expected error code in {_VALID_ERROR_CODES}, got {result['error']!r}"
    )
    assert isinstance(result["message"], str) and result["message"], (
        "expected non-empty message string"
    )


def _safe_invoke(tool: Any, payload: dict[str, Any]) -> Any:
    """Invoke ``tool`` with ``payload``; any exception bubbles up as failure.

    LangChain's ``BaseTool.invoke`` wraps the underlying function and can
    convert internal raises into ``ToolException``. Property 6 requires
    that no exception escapes either layer for unknown category/intent
    values.
    """

    return tool.invoke(payload)


# ---------------------------------------------------------------------------
# Property 6 -- tools return structured errors for missing values
# ---------------------------------------------------------------------------

@given(df=dataframes_st(), data=st.data())
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_tools_return_structured_errors_for_missing_values(
    monkeypatch: pytest.MonkeyPatch,
    df: pd.DataFrame,
    data: st.DataObject,
) -> None:
    """Feature: customer-service-data-analyst-agent, Property 6: Tools return structured errors for missing values.

    Validates Requirements 3.8.
    """

    bogus = data.draw(non_existent_string_st(df))

    # Safety net: ensure no test path can reach a real Nebius LLM. The
    # tool checks the category first and short-circuits before invoking
    # the model, but we patch defensively so a regression that flips
    # that ordering would still fail loudly without making a network
    # call.
    fake_llm = FakeChatModel(invoke_response=type("R", (), {"content": "stub"})())

    def _fake_get_llm(*_a: Any, **_kw: Any) -> FakeChatModel:
        return fake_llm

    monkeypatch.setattr("csa_agent.llm.get_llm", _fake_get_llm)
    monkeypatch.setattr("csa_agent.tools.core.get_llm", _fake_get_llm)

    tools = _tools_by_name(df)

    # ------------------------------------------------------------------
    # Tools that take a required category/intent parameter.
    # ------------------------------------------------------------------
    required_param_cases: list[tuple[str, dict[str, Any]]] = [
        ("filter_by_intent", {"intent": bogus}),
        ("filter_by_category", {"category": bogus}),
        ("get_intent_distribution", {"category": bogus}),
        ("summarize_category", {"category": bogus}),
    ]
    for tool_name, payload in required_param_cases:
        result = _safe_invoke(tools[tool_name], payload)
        _assert_tool_error(result, bogus)

    # ------------------------------------------------------------------
    # Tools with optional filters: exercise both parameter slots so we
    # cover the category-not-found and intent-not-found branches.
    # ------------------------------------------------------------------
    optional_param_cases: list[tuple[str, dict[str, Any]]] = [
        ("count_rows", {"category": bogus}),
        ("count_rows", {"intent": bogus}),
        ("show_examples", {"category": bogus, "n": 5}),
        ("show_examples", {"intent": bogus, "n": 5}),
    ]
    for tool_name, payload in optional_param_cases:
        result = _safe_invoke(tools[tool_name], payload)
        _assert_tool_error(result, bogus)

    # The summarize path must never reach the LLM for a bogus category
    # (the existence check short-circuits earlier). This guard protects
    # future refactors from regressing the ordering.
    assert fake_llm.invocations == [], (
        "summarize_category should not call the LLM when the category is bogus"
    )
