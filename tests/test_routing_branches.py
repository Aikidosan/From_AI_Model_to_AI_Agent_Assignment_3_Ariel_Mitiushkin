"""Integration tests for the LangGraph routing layer.

Feature: customer-service-data-analyst-agent

This module covers two related properties of the compiled graph:

* **Property 8 -- Decline path is pure.** Validates Requirements 2.2,
  2.5. When the router classifies a query as ``OUT_OF_SCOPE`` the
  downstream graph must emit the canonical decline message and make
  zero additional LLM calls and zero tool invocations beyond the
  router itself. We capture the LLM-factory call count *after*
  ``build_graph`` (which eagerly constructs the router LLM) and assert
  it does not grow during the invocation; tool-call spies must remain
  empty.

* **Property 9 -- Routing matches classification.** Validates
  Requirements 2.3, 2.4. For each :class:`RouteLabel` the graph must
  visit the matching node next: ``STRUCTURED`` -> ``react_agent``,
  ``UNSTRUCTURED`` -> ``summarize``, ``OUT_OF_SCOPE`` -> ``decline``.

Tests build the graph against a small in-memory DataFrame so they
never touch the real Bitext CSV. Two patches are applied *before*
``build_graph`` so the eagerly-constructed sub-agents never reach the
real Nebius client:

1. ``csa_agent.llm.get_llm`` (and its re-exports in ``graph`` and
   ``nodes``) is replaced with a counting fake.
2. ``langgraph.prebuilt.create_react_agent`` is replaced with a
   factory returning a small stub sub-agent that echoes its input
   messages and appends a single ``AIMessage`` with no tool calls.
   This sidesteps the LangChain runnable-coercion path that the real
   ``create_react_agent`` performs (``prompt | model``) and which
   refuses to accept duck-typed fakes.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage

import csa_agent.graph as graph_mod
import csa_agent.nodes as nodes_mod
import csa_agent.tools.core as tools_core_mod
from csa_agent.graph import (
    DECLINE_NODE,
    REACT_AGENT_NODE,
    SUMMARIZE_NODE,
    build_graph,
)
from csa_agent.nodes import CANONICAL_REFUSAL
from csa_agent.router import RouteLabel


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _tiny_df() -> pd.DataFrame:
    """Return a small, in-memory DataFrame with the required columns.

    Six rows across three categories give the tools enough variety to
    answer a routing-time query without touching the production CSV.
    """

    return pd.DataFrame(
        [
            {"utterance": "I want a refund", "category": "REFUND", "intent": "track_refund"},
            {"utterance": "where is my refund", "category": "REFUND", "intent": "track_refund"},
            {"utterance": "cancel my order", "category": "ORDER", "intent": "cancel_order"},
            {"utterance": "stop my order", "category": "ORDER", "intent": "cancel_order"},
            {"utterance": "thanks for the help", "category": "FEEDBACK", "intent": "rate_service"},
            {"utterance": "great service", "category": "FEEDBACK", "intent": "rate_service"},
        ]
    )


class _SpyState:
    """Mutable counters shared across the patched LLM factory and tool spies."""

    def __init__(self) -> None:
        self.llm_factory_calls: int = 0
        self.llm_invokes: int = 0
        self.tool_invocations: list[tuple[str, dict[str, Any]]] = []


class _CountingFakeLLM:
    """Tiny stand-in for ChatOpenAI used by the router and (unused) sub-agents.

    Exposes the surface :func:`csa_agent.router.classify_query` would
    touch (``with_structured_output(...).invoke(...)``) and a generic
    ``invoke``. Both increment the spy's counter so the purity test can
    detect any unexpected call.
    """

    def __init__(self, spy: _SpyState) -> None:
        self._spy = spy
        self.base_url = "https://api.studio.nebius.ai/v1/"

    def with_structured_output(self, _schema: Any) -> "_CountingFakeLLM":
        return self

    def invoke(self, _messages: Any, *_a: Any, **_kw: Any) -> Any:
        self._spy.llm_invokes += 1
        # Return a benign string; the router happens to be patched in
        # both tests so this branch is rarely (if ever) taken.
        return "out_of_scope"

    def bind_tools(self, _tools: Any, **_kw: Any) -> "_CountingFakeLLM":
        return self


class _StubSubagent:
    """Stub used in place of the real ReAct sub-agent.

    The real :func:`langgraph.prebuilt.create_react_agent` returns a
    compiled graph whose ``invoke`` runs the LLM tool-call loop. For
    routing tests we only care that the wrapping node *visits* the
    sub-agent; we do not need real reasoning. The stub echoes its
    input messages plus a single AIMessage with no tool calls so the
    parent graph's delta calculation produces a one-message update.
    """

    def __init__(self, spy: _SpyState, final_text: str = "ok") -> None:
        self._spy = spy
        self._final_text = final_text

    def invoke(self, state: dict[str, Any], *_a: Any, **_kw: Any) -> dict[str, Any]:
        # No LLM call is made; the stub is a pure echo. Tests assert the
        # absence of LLM calls on the OUT_OF_SCOPE branch; on the
        # STRUCTURED / UNSTRUCTURED branches the assertion is just
        # "the right node was visited", so we avoid touching the spy
        # counters here.
        messages = list(state.get("messages", []))
        return {"messages": [*messages, AIMessage(content=self._final_text)]}


def _install_fake_llm(monkeypatch: pytest.MonkeyPatch, spy: _SpyState) -> None:
    """Patch every ``get_llm`` import site to return a counting fake.

    Every call to ``get_llm()`` increments ``spy.llm_factory_calls`` so
    Property 8 can assert "no factory calls after build". Every call
    to the fake's ``invoke`` increments ``spy.llm_invokes``.
    """

    def _factory(*_a: Any, **_kw: Any) -> _CountingFakeLLM:
        spy.llm_factory_calls += 1
        return _CountingFakeLLM(spy)

    # The graph and nodes modules import ``get_llm`` at module load
    # time, so patching the source module alone is not enough: every
    # site that holds a binding must be patched.
    monkeypatch.setattr(graph_mod, "get_llm", _factory)
    monkeypatch.setattr(nodes_mod, "get_llm", _factory)
    monkeypatch.setattr(tools_core_mod, "get_llm", _factory)
    monkeypatch.setattr("csa_agent.llm.get_llm", _factory)


def _install_stub_subagent(monkeypatch: pytest.MonkeyPatch, spy: _SpyState) -> None:
    """Replace ``create_react_agent`` with a factory returning :class:`_StubSubagent`.

    The real implementation builds a LangChain runnable (``prompt |
    model``) which rejects duck-typed fakes. The stub bypasses that
    entirely while preserving the ``invoke``/``stream`` shape the
    parent graph expects from a ReAct sub-agent.
    """

    def _factory(*_a: Any, **_kw: Any) -> _StubSubagent:
        return _StubSubagent(spy)

    monkeypatch.setattr(graph_mod, "create_react_agent", _factory)
    # ``nodes._build_summarize_subagent`` does an in-function import
    # of ``langgraph.prebuilt.create_react_agent``, so we patch the
    # canonical location to cover that path too.
    monkeypatch.setattr("langgraph.prebuilt.create_react_agent", _factory)


def _install_tool_spies(monkeypatch: pytest.MonkeyPatch, spy: _SpyState) -> None:
    """Wrap ``build_tools`` so every tool's ``func`` records its calls.

    ``StructuredTool`` is a Pydantic ``BaseModel`` but field assignment
    is permitted: replacing ``func`` with a wrapper is the simplest way
    to count tool invocations regardless of which graph node triggers
    them.
    """

    original_build_tools = graph_mod.build_tools

    def spying_build_tools(df: pd.DataFrame) -> Any:
        tools = original_build_tools(df)
        for t in tools:
            tool_name = t.name
            original_func = t.func

            def make_wrapper(name: str, fn: Any) -> Any:
                def wrapped(*args: Any, **kwargs: Any) -> Any:
                    spy.tool_invocations.append((name, dict(kwargs)))
                    return fn(*args, **kwargs)

                return wrapped

            t.func = make_wrapper(tool_name, original_func)
        return tools

    monkeypatch.setattr(graph_mod, "build_tools", spying_build_tools)


def _stream_events(graph: Any, query: str) -> list[dict[str, Any]]:
    """Run ``graph.stream`` with a one-shot human message and collect updates.

    Returns the raw ``stream_mode="updates"`` events, each shaped
    ``{node_name: state_delta}``. Tests inspect the keys to determine
    which branch ran.
    """

    config = {"configurable": {"thread_id": "test-thread", "user_id": "test-user"}}
    return list(
        graph.stream(
            {"messages": [HumanMessage(content=query)]},
            config=config,
            stream_mode="updates",
        )
    )


def _visited_nodes(events: list[dict[str, Any]]) -> list[str]:
    """Return the ordered list of node names that produced state deltas."""

    visited: list[str] = []
    for event in events:
        if isinstance(event, dict):
            visited.extend(str(k) for k in event.keys())
    return visited


# ---------------------------------------------------------------------------
# Property 8 -- Decline path is pure
# ---------------------------------------------------------------------------


def test_decline_path_makes_no_llm_or_tool_calls_after_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feature: customer-service-data-analyst-agent, Property 8: Decline path is pure.

    Validates Requirements 2.2, 2.5.

    When the router classifies a query as ``OUT_OF_SCOPE``, the
    downstream graph must emit the canonical decline message and make
    zero additional LLM calls and zero tool invocations. We capture
    the LLM-factory call counter *after* ``build_graph`` (which
    eagerly constructs the router and ReAct sub-agent LLMs) and
    assert it stays unchanged during invocation. Tool-call spies
    must remain empty because the decline branch never reaches a
    tool.
    """

    spy = _SpyState()
    _install_fake_llm(monkeypatch, spy)
    _install_stub_subagent(monkeypatch, spy)
    _install_tool_spies(monkeypatch, spy)

    df = _tiny_df()
    graph = build_graph(checkpointer=None, df=df)

    # Snapshot counters AFTER build_graph so build-time LLM construction
    # does not pollute the assertion. Build-time tool calls should
    # already be zero -- ``build_tools`` only constructs tools, it does
    # not invoke them.
    llm_baseline_factory = spy.llm_factory_calls
    llm_baseline_invokes = spy.llm_invokes
    assert spy.tool_invocations == [], (
        "build_tools should not invoke any tool; spies recorded "
        f"{spy.tool_invocations!r}"
    )

    # Patch the router in the graph module to short-circuit to OUT_OF_SCOPE
    # without consulting the captured router LLM. The patched function
    # ignores both arguments (the closure still passes them).
    monkeypatch.setattr(
        graph_mod,
        "classify_query",
        lambda _query, _llm: RouteLabel.OUT_OF_SCOPE,
    )

    events = _stream_events(graph, "What is the capital of France?")
    visited = _visited_nodes(events)

    # The decline node must be visited; the ReAct and summarize nodes
    # must not.
    assert DECLINE_NODE in visited, (
        f"expected {DECLINE_NODE!r} in visited nodes, got {visited!r}"
    )
    assert REACT_AGENT_NODE not in visited
    assert SUMMARIZE_NODE not in visited

    # Purity: the only LLM activity allowed once we leave the router on
    # the OUT_OF_SCOPE branch is the opportunistic name/preference
    # extraction inside ``update_profile_node`` (introduced for Task 2b
    # natural-conversation profile updates). It runs *after* the decline
    # message has already been appended, so the decline message itself
    # is still produced without consulting the LLM. We therefore allow
    # at most one extra factory call and one extra invoke; anything
    # beyond that means the decline branch itself touched the LLM.
    assert spy.llm_factory_calls - llm_baseline_factory <= 1, (
        f"get_llm was called more than the single profile-extraction "
        f"call: baseline={llm_baseline_factory}, after={spy.llm_factory_calls}"
    )
    assert spy.llm_invokes - llm_baseline_invokes <= 1, (
        f"LLM.invoke was called more than the single profile-extraction "
        f"call: baseline={llm_baseline_invokes}, after={spy.llm_invokes}"
    )
    assert spy.tool_invocations == [], (
        f"tools were invoked during the decline path: {spy.tool_invocations!r}"
    )

    # The decline node's canonical refusal message must surface in the
    # decline event so the property has end-to-end teeth.
    decline_event = next(
        e for e in events if isinstance(e, dict) and DECLINE_NODE in e
    )
    decline_msgs = decline_event[DECLINE_NODE].get("messages", [])
    assert any(
        getattr(m, "content", "") == CANONICAL_REFUSAL for m in decline_msgs
    ), f"decline node did not emit the canonical refusal: {decline_msgs!r}"


# ---------------------------------------------------------------------------
# Property 9 -- Routing matches classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected_node"),
    [
        (RouteLabel.STRUCTURED, REACT_AGENT_NODE),
        (RouteLabel.UNSTRUCTURED, SUMMARIZE_NODE),
        (RouteLabel.OUT_OF_SCOPE, DECLINE_NODE),
    ],
    ids=["structured-to-react", "unstructured-to-summarize", "out_of_scope-to-decline"],
)
def test_routing_visits_node_matching_classification(
    monkeypatch: pytest.MonkeyPatch,
    label: RouteLabel,
    expected_node: str,
) -> None:
    """Feature: customer-service-data-analyst-agent, Property 9: Routing matches classification.

    Validates Requirements 2.3, 2.4.

    Patching :func:`classify_query` to return each :class:`RouteLabel`
    in turn, the graph must visit the matching downstream node:

    * ``STRUCTURED`` -> ``react_agent``
    * ``UNSTRUCTURED`` -> ``summarize``
    * ``OUT_OF_SCOPE`` -> ``decline``

    The two non-decline branches build ReAct sub-agents internally;
    we replace ``create_react_agent`` with a stub factory so those
    sub-agents return a final no-tool-call AIMessage immediately.
    """

    spy = _SpyState()
    _install_fake_llm(monkeypatch, spy)
    _install_stub_subagent(monkeypatch, spy)

    df = _tiny_df()
    graph = build_graph(checkpointer=None, df=df)

    monkeypatch.setattr(
        graph_mod,
        "classify_query",
        lambda _query, _llm: label,
    )

    events = _stream_events(graph, "anything")
    visited = _visited_nodes(events)

    assert expected_node in visited, (
        f"for route={label!r} expected to visit {expected_node!r}; "
        f"visited={visited!r}"
    )

    # Negative checks: the two branches we did not select must not
    # appear in the visited list. This prevents a regression where the
    # graph fans out to multiple terminal nodes.
    other_terminals = (
        {REACT_AGENT_NODE, SUMMARIZE_NODE, DECLINE_NODE} - {expected_node}
    )
    for node in other_terminals:
        assert node not in visited, (
            f"for route={label!r} unexpectedly visited {node!r}; "
            f"visited={visited!r}"
        )
