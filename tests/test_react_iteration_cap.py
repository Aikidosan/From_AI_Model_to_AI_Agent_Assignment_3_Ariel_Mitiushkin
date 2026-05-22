"""Property test for the ReAct agent iteration cap.

Feature: customer-service-data-analyst-agent
Property 10: ReAct agent terminates within the iteration cap.

Validates: Requirements 4.2, 4.3.

For *any* user query and *for any* mocked LLM behaviour that always
emits a tool call, the ReAct agent terminates after at most
``settings.max_iterations`` reasoning iterations and the final state
contains a non-empty user-visible message (no unhandled exception, no
infinite loop).

Two complementary tests cover the property:

* :func:`test_react_terminates_with_non_empty_fallback_when_cap_reached`
  -- replaces the ReAct sub-agent with a stub that raises
  :class:`langgraph.errors.GraphRecursionError` from ``invoke``,
  simulating the recursion-limit signal LangGraph emits when the cap
  is reached. The graph's ``react_agent_node`` must catch that error
  and emit the graceful fallback :class:`AIMessage` required by
  Requirement 4.3, producing a non-empty final user-visible message.
* :func:`test_react_passes_settings_max_iterations_as_recursion_limit`
  -- replaces the sub-agent with a stub that records the ``config``
  it was invoked with, then asserts the graph passes
  ``recursion_limit == settings.max_iterations`` on every invocation.
  This is the mechanism the cap is implemented by, so verifying it
  closes the loop on Requirement 4.2.

Together the two tests show: the cap value is wired from settings
into the sub-agent invocation (Test 2), and when the sub-agent
actually hits the cap the graph terminates with a non-empty fallback
message (Test 1).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings as hsettings
from hypothesis import strategies as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

import csa_agent.graph as graph_mod
import csa_agent.nodes as nodes_mod
import csa_agent.tools.core as tools_core_mod
from csa_agent.config import (
    DEFAULT_CHECKPOINT_DB,
    DEFAULT_DATASET_PATH,
    DEFAULT_NEBIUS_BASE_URL,
    DEFAULT_NEBIUS_MODEL,
    DEFAULT_PROFILE_DIR,
    Settings,
)
from csa_agent.graph import build_graph
from csa_agent.router import RouteLabel


# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------


def _tiny_df() -> pd.DataFrame:
    """Return a minimal DataFrame so build_tools succeeds without the real CSV."""

    return pd.DataFrame(
        [
            {
                "utterance": "I want a refund",
                "category": "REFUND",
                "intent": "track_refund",
            },
        ]
    )


class _AlwaysRecurseSubagent:
    """Stub ReAct sub-agent that simulates hitting the recursion cap.

    Always raises :class:`GraphRecursionError` from ``invoke``; this is
    the exact signal LangGraph emits when the configured
    ``recursion_limit`` is reached. The graph's ``react_agent_node`` is
    expected to catch the error and emit the canonical fallback
    AIMessage.
    """

    def __init__(self) -> None:
        self.last_config: dict[str, Any] | None = None
        self.invocations: int = 0

    def invoke(self, _state: Any, config: dict[str, Any] | None = None, **_kw: Any) -> Any:
        self.last_config = config
        self.invocations += 1
        raise GraphRecursionError(
            "Recursion limit reached (simulated by test stub)"
        )


class _CaptureRecursionLimitSubagent:
    """Stub ReAct sub-agent that records ``config`` and returns a final AIMessage.

    Used to verify that the graph passes ``recursion_limit`` derived
    from ``settings.max_iterations`` to the sub-agent's ``invoke`` call.
    """

    def __init__(self) -> None:
        self.last_config: dict[str, Any] | None = None
        self.invocations: int = 0

    def invoke(
        self,
        state: dict[str, Any],
        config: dict[str, Any] | None = None,
        **_kw: Any,
    ) -> dict[str, Any]:
        self.last_config = config
        self.invocations += 1
        msgs = list(state.get("messages", []))
        return {"messages": [*msgs, AIMessage(content="ok")]}


class _CountingFakeLLM:
    """Tiny stand-in for ChatOpenAI used by the router and tool factories.

    Exposes the surface ``classify_query`` touches
    (``with_structured_output(...).invoke(...)``) plus the generic
    ``invoke`` and ``bind_tools`` methods other code paths may call.
    """

    base_url = "https://api.studio.nebius.ai/v1/"

    def with_structured_output(self, _schema: Any) -> "_CountingFakeLLM":
        return self

    def invoke(self, *_a: Any, **_kw: Any) -> str:
        return "structured"

    def bind_tools(self, *_a: Any, **_kw: Any) -> "_CountingFakeLLM":
        return self


def _install_fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch every ``get_llm`` import site to return the same counting fake.

    The graph and nodes modules import ``get_llm`` at module load time,
    so patching the source module alone is not enough: every site that
    holds a binding must be patched.
    """

    fake = _CountingFakeLLM()

    def _factory(*_a: Any, **_kw: Any) -> _CountingFakeLLM:
        return fake

    monkeypatch.setattr(graph_mod, "get_llm", _factory)
    monkeypatch.setattr(nodes_mod, "get_llm", _factory)
    monkeypatch.setattr(tools_core_mod, "get_llm", _factory)
    monkeypatch.setattr("csa_agent.llm.get_llm", _factory)


def _install_subagent(
    monkeypatch: pytest.MonkeyPatch, subagent: Any
) -> None:
    """Replace ``create_react_agent`` with a factory returning ``subagent``.

    Patches both ``csa_agent.graph.create_react_agent`` (which
    ``_make_react_agent_node`` resolves) and ``langgraph.prebuilt.create_react_agent``
    (which ``nodes._build_summarize_subagent`` does an in-function
    import of) so neither path constructs a real ReAct agent.
    """

    def _factory(*_a: Any, **_kw: Any) -> Any:
        return subagent

    monkeypatch.setattr(graph_mod, "create_react_agent", _factory)
    monkeypatch.setattr("langgraph.prebuilt.create_react_agent", _factory)


def _route_to_react(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the router so every query is classified as STRUCTURED."""

    monkeypatch.setattr(
        graph_mod,
        "classify_query",
        lambda _query, _llm: RouteLabel.STRUCTURED,
    )


def _make_settings(max_iter: int) -> Settings:
    """Build a Settings instance carrying a custom ``max_iterations``.

    Other fields are filled with the module-level defaults so the
    Pydantic model validates without touching the environment.
    """

    return Settings(
        nebius_api_key="stub-test-key",
        nebius_base_url=DEFAULT_NEBIUS_BASE_URL,
        nebius_model=DEFAULT_NEBIUS_MODEL,
        dataset_path=DEFAULT_DATASET_PATH,
        checkpoint_db=DEFAULT_CHECKPOINT_DB,
        profile_dir=DEFAULT_PROFILE_DIR,
        max_iterations=max_iter,
    )


# ---------------------------------------------------------------------------
# Property 10 -- non-empty fallback on cap reached
# ---------------------------------------------------------------------------


@given(query=st.text(min_size=0, max_size=80))
@hsettings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_react_terminates_with_non_empty_fallback_when_cap_reached(
    query: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feature: customer-service-data-analyst-agent, Property 10: ReAct agent terminates within the iteration cap.

    Validates Requirements 4.2, 4.3.

    For an arbitrary user query, when the ReAct sub-agent always
    raises :class:`GraphRecursionError` (the LangGraph signal that the
    recursion limit has been reached), the outer graph must:

    * terminate without propagating the error to the caller;
    * emit a non-empty user-visible AIMessage as the final message;
    * the final message must read like the graceful fallback required
      by Requirement 4.3 (mentioning the iteration / step limit).
    """

    _install_fake_llm(monkeypatch)
    stub = _AlwaysRecurseSubagent()
    _install_subagent(monkeypatch, stub)
    _route_to_react(monkeypatch)

    graph = build_graph(checkpointer=None, df=_tiny_df())

    # Hypothesis can generate the empty string; ensure the HumanMessage
    # carries some content so downstream nodes that pull "the latest
    # human text" do not see an empty payload. The router is patched to
    # ignore the query, so any non-empty value works.
    safe_query = query if query else "x"
    config = {"configurable": {"thread_id": "t-cap", "user_id": "u-cap"}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=safe_query)]},
        config=config,
    )

    # The stub must actually have been invoked (sanity check that we
    # routed to the ReAct branch and the cap was triggered).
    assert stub.invocations >= 1, "ReAct sub-agent stub was never invoked"

    # Termination: the result must contain a final AIMessage with
    # non-empty content. Filter to AIMessages that have no pending
    # tool_calls -- those represent user-visible final answers.
    messages = list(result.get("messages", []))
    user_visible_ais = [
        m
        for m in messages
        if isinstance(m, AIMessage)
        and isinstance(m.content, str)
        and m.content.strip()
        and not (getattr(m, "tool_calls", None) or [])
    ]
    assert user_visible_ais, (
        f"no non-empty user-visible AIMessage in graph output; messages={messages!r}"
    )

    final_text = user_visible_ais[-1].content
    assert final_text, "final user-visible AIMessage is empty"

    # The fallback wording is documented in graph.py as
    # "I couldn't complete this query within the reasoning step limit."
    # Assert on the substantive keywords rather than the exact phrasing
    # so minor wording tweaks do not break the test.
    lowered = final_text.lower()
    assert ("limit" in lowered) or ("step" in lowered) or ("couldn" in lowered), (
        f"final message does not look like the iteration-cap fallback: {final_text!r}"
    )


# ---------------------------------------------------------------------------
# Property 10 -- recursion_limit derives from settings.max_iterations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_iter", [1, 4, 8, 15])
def test_react_passes_settings_max_iterations_as_recursion_limit(
    max_iter: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feature: customer-service-data-analyst-agent, Property 10: ReAct agent terminates within the iteration cap.

    Validates Requirements 4.2, 4.3.

    The graph's ``react_agent_node`` must invoke the ReAct sub-agent
    with ``config["recursion_limit"]`` equal to
    ``settings.max_iterations``. This is the mechanism that *causes*
    LangGraph to raise :class:`GraphRecursionError` after the
    configured number of super-steps; verifying the wiring closes the
    loop on Requirement 4.2.
    """

    # Patch get_settings BEFORE build_graph because
    # ``_make_react_agent_node`` captures the Settings instance in a
    # closure at build time.
    fake_settings = _make_settings(max_iter)
    monkeypatch.setattr(graph_mod, "get_settings", lambda: fake_settings)

    _install_fake_llm(monkeypatch)
    stub = _CaptureRecursionLimitSubagent()
    _install_subagent(monkeypatch, stub)
    _route_to_react(monkeypatch)

    graph = build_graph(checkpointer=None, df=_tiny_df())

    config = {"configurable": {"thread_id": "t-cfg", "user_id": "u-cfg"}}
    graph.invoke(
        {"messages": [HumanMessage(content="anything")]},
        config=config,
    )

    assert stub.invocations >= 1, "ReAct sub-agent stub was never invoked"
    assert stub.last_config is not None, "no config was passed to the sub-agent"
    actual_limit = stub.last_config.get("recursion_limit")
    assert actual_limit == max_iter, (
        f"expected recursion_limit={max_iter} (from settings.max_iterations), "
        f"got {actual_limit!r}"
    )
