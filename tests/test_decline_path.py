"""Integration test for the decline-path purity invariant.

Feature: customer-service-data-analyst-agent
Property 8 — Decline path is pure.
Validates: Requirements 2.2, 2.5.

When the router classifies a query as :attr:`RouteLabel.OUT_OF_SCOPE`,
the downstream graph must:

* emit the canonical refusal :class:`AIMessage` (Requirement 2.2), and
* make zero LLM calls and zero tool invocations after the router
  classification step (Requirement 2.5).

Test design:

* Patch :func:`csa_agent.llm.get_llm` (and every site that imported it
  at module load time) with a :class:`MagicMock` factory so any LLM
  construction is observable and no real Nebius client is created.
* Stub :func:`langgraph.prebuilt.create_react_agent` so the ReAct
  sub-agent built eagerly inside :func:`build_graph` does not try to
  wrap our duck-typed mock in a real LangChain runnable.
* Replace :func:`csa_agent.tools.core.build_tools` with a factory that
  takes the *real* tool list and replaces each tool's underlying
  callable with a :class:`MagicMock`. The decline path must invoke
  none of them.
* Snapshot the ``get_llm`` factory call count *after* :func:`build_graph`
  (which legitimately constructs the router and ReAct LLM clients at
  build time) and assert no further calls occur during the run.
* Patch :func:`csa_agent.router.classify_query` so the router itself
  does not consult the captured LLM — that lets us assert the
  classifier LLM mock's ``invoke`` and ``with_structured_output`` are
  never called either.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage

import csa_agent.graph as graph_mod
import csa_agent.nodes as nodes_mod
import csa_agent.tools.core as tools_core_mod
from csa_agent.graph import build_graph
from csa_agent.nodes import CANONICAL_REFUSAL
from csa_agent.router import RouteLabel
from csa_agent.tools.core import build_tools as _real_build_tools


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tiny_df() -> pd.DataFrame:
    """A minimal DataFrame with the required columns.

    The decline path never reads any rows, but :func:`build_tools`
    inspects the columns when constructing the tools, so we need a
    well-formed frame.
    """

    return pd.DataFrame(
        [
            {"utterance": "i want a refund", "category": "REFUND", "intent": "track_refund"},
            {"utterance": "cancel my order", "category": "ORDER", "intent": "cancel_order"},
        ]
    )


class _StubSubagent:
    """Minimal stub for the ReAct sub-agent built by ``create_react_agent``.

    The decline path never reaches the sub-agent, but the parent graph
    constructs one eagerly inside ``_make_react_agent_node``. The stub
    makes the construction succeed without touching a real LLM.
    """

    def __init__(self) -> None:
        self.invoke = MagicMock(return_value={"messages": []})


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_decline_path_invokes_no_llm_and_no_tools(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Property 8: decline path emits the refusal and stays pure.

    Validates Requirements 2.2, 2.5.

    Walks the graph end-to-end with the router patched to
    ``OUT_OF_SCOPE`` and asserts:

    * The final state contains an :class:`AIMessage` with the canonical
      refusal text.
    * The patched :func:`get_llm` mock is *not* called during the run
      (build-time calls are excluded by snapshotting after
      :func:`build_graph`).
    * The classifier LLM mock returned by ``get_llm`` is never invoked
      (because :func:`classify_query` is also stubbed).
    * No tool ``MagicMock`` is invoked at any point.
    """

    # Keep profile writes hermetic: write any update_profile_node output
    # under ``tmp_path`` rather than the repo's ``profiles/`` directory.
    monkeypatch.setenv("PROFILE_DIR", str(tmp_path))

    # ---------------- LLM factory: mock every call site ------------------
    # ``llm_mock`` stands in for the real Nebius ``ChatOpenAI`` client.
    # Wiring ``with_structured_output`` to return ``llm_mock`` itself lets
    # the chained call ``llm.with_structured_output(...).invoke(...)``
    # land back on the same mock so we can assert it was never used.
    llm_mock = MagicMock(name="ChatOpenAI")
    llm_mock.with_structured_output.return_value = llm_mock
    llm_mock.invoke.return_value = "out_of_scope"

    get_llm_mock = MagicMock(name="get_llm", return_value=llm_mock)

    # Patch every binding that resolved ``get_llm`` at module load time.
    monkeypatch.setattr(graph_mod, "get_llm", get_llm_mock)
    monkeypatch.setattr(nodes_mod, "get_llm", get_llm_mock)
    monkeypatch.setattr(tools_core_mod, "get_llm", get_llm_mock)
    monkeypatch.setattr("csa_agent.llm.get_llm", get_llm_mock)

    # ---------------- ReAct sub-agent: stub ``create_react_agent`` -------
    # ``build_graph`` eagerly constructs the ReAct sub-agent. The real
    # implementation builds a LangChain runnable that rejects duck-typed
    # mocks; the stub keeps construction inert.
    stub_subagent = _StubSubagent()

    def _stub_create_react_agent(*_args: Any, **_kwargs: Any) -> _StubSubagent:
        return stub_subagent

    monkeypatch.setattr(graph_mod, "create_react_agent", _stub_create_react_agent)
    monkeypatch.setattr(
        "langgraph.prebuilt.create_react_agent", _stub_create_react_agent
    )

    # ---------------- Tools: replace each ``.func`` with a MagicMock -----
    # ``build_tools`` returns ``StructuredTool`` instances whose ``func``
    # field is the actual callable. Replacing it with a MagicMock makes
    # any invocation observable.
    tool_mocks: dict[str, MagicMock] = {}

    def _spy_build_tools(df: pd.DataFrame) -> list[Any]:
        tools = _real_build_tools(df)
        for t in tools:
            mock = MagicMock(name=f"tool_func::{t.name}", return_value=[])
            tool_mocks[t.name] = mock
            t.func = mock
        return tools

    monkeypatch.setattr(graph_mod, "build_tools", _spy_build_tools)
    monkeypatch.setattr(tools_core_mod, "build_tools", _spy_build_tools)

    # ---------------- Build the graph ------------------------------------
    df = _tiny_df()
    graph = build_graph(checkpointer=None, df=df)

    # Sanity: build_tools should not invoke any tool.
    invoked_during_build = {
        name: m.call_count for name, m in tool_mocks.items() if m.call_count
    }
    assert not invoked_during_build, (
        f"tools were invoked during build_graph: {invoked_during_build!r}"
    )

    # Snapshot LLM-factory calls *after* build_graph: build legitimately
    # constructs the router classifier LLM and the ReAct sub-agent's LLM,
    # both via ``get_llm``. Property 8 is about purity *after* the router
    # classification, not at construction time.
    get_llm_mock.reset_mock()
    llm_mock.reset_mock()
    # ``with_structured_output`` is an attribute of ``llm_mock``; resetting
    # the parent also resets its child mocks, but we re-arm the chain so
    # any accidental call still routes back to a known sentinel.
    llm_mock.with_structured_output.return_value = llm_mock
    llm_mock.invoke.return_value = "out_of_scope"

    # ---------------- Patch the router to OUT_OF_SCOPE -------------------
    # Patching ``classify_query`` short-circuits the router so even the
    # captured classifier LLM mock is never consulted, satisfying the
    # cleaner "get_llm was never called at all" reading of Property 8.
    classify_calls: list[tuple[str, Any]] = []

    def _stub_classify(user_query: str, llm: Any) -> RouteLabel:
        classify_calls.append((user_query, llm))
        return RouteLabel.OUT_OF_SCOPE

    monkeypatch.setattr(graph_mod, "classify_query", _stub_classify)

    # ---------------- Run the graph --------------------------------------
    config: dict[str, Any] = {
        "configurable": {"thread_id": "decline-thread", "user_id": "decline-user"},
    }
    final_state = graph.invoke(
        {"messages": [HumanMessage(content="What is the capital of France?")]},
        config=config,
    )

    # ---------------- Assertions -----------------------------------------

    # The router was visited exactly once.
    assert len(classify_calls) == 1, (
        f"classify_query should run exactly once, got {len(classify_calls)} "
        f"call(s): {classify_calls!r}"
    )

    # 1. Final state contains the canonical refusal.
    final_messages = list(final_state.get("messages", []))
    refusal_messages = [
        m
        for m in final_messages
        if isinstance(m, AIMessage) and m.content == CANONICAL_REFUSAL
    ]
    assert refusal_messages, (
        "expected an AIMessage with the canonical refusal text in the final "
        f"state; got: {final_messages!r}"
    )

    # 2. ``get_llm`` was not called after the router classification.
    assert get_llm_mock.call_count == 0, (
        "get_llm was invoked during the decline path after the router "
        f"classification (count={get_llm_mock.call_count})"
    )

    # 3. The captured router-classifier LLM was never consulted because
    #    classify_query was stubbed.
    assert llm_mock.with_structured_output.call_count == 0, (
        "router classifier LLM.with_structured_output was called "
        f"({llm_mock.with_structured_output.call_count} times)"
    )
    assert llm_mock.invoke.call_count == 0, (
        f"router classifier LLM.invoke was called ({llm_mock.invoke.call_count} times)"
    )

    # 4. No tool MagicMock was invoked at any point during the run.
    invoked_tools = {
        name: m.call_count for name, m in tool_mocks.items() if m.call_count
    }
    assert not invoked_tools, (
        f"tools were invoked during the decline path: {invoked_tools!r}"
    )

    # 5. The ReAct sub-agent stub was never reached (defense in depth: the
    #    decline branch must not fan out to the ReAct node).
    assert stub_subagent.invoke.call_count == 0, (
        "ReAct sub-agent was invoked during the decline path "
        f"(count={stub_subagent.invoke.call_count})"
    )
