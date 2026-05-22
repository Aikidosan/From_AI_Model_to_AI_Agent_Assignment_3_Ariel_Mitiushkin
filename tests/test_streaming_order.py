"""Property test for the streaming-event ordering contract.

Feature: customer-service-data-analyst-agent
Property 11: Tool calls are streamed before the final answer.

Validates: Requirements 4.4, 5.4, 11.3.

For *any* run that produces at least one tool call, every tool-call
event index in :func:`csa_agent.graph.stream_graph`'s output is
strictly less than the final-answer event index. This is the
ordering invariant that lets the CLI and the Streamlit UI render
reasoning steps before the user sees the final answer.

The test patches the ReAct sub-agent with a stub that returns a
canned message sequence containing one or more ``AIMessage`` tool
calls paired with ``ToolMessage`` observations, followed by a final
``AIMessage`` with no pending tool calls. The Hypothesis strategy
generates the names, arguments, and observations of the tool calls
(plus the final answer text) so the property is checked across a
broad range of payload shapes -- not just one hard-coded fixture.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings as hsettings
from hypothesis import strategies as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

import csa_agent.graph as graph_mod
import csa_agent.nodes as nodes_mod
import csa_agent.tools.core as tools_core_mod
from csa_agent.graph import build_graph, stream_graph
from csa_agent.router import RouteLabel


# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------


def _tiny_df() -> pd.DataFrame:
    """Return a minimal DataFrame so build_tools succeeds without the real CSV."""

    return pd.DataFrame(
        [
            {
                "utterance": "hi",
                "category": "REFUND",
                "intent": "track_refund",
            },
        ]
    )


class _CountingFakeLLM:
    """Tiny stand-in for ChatOpenAI used by the router and sub-agent factory.

    Exposes the surface :func:`csa_agent.router.classify_query` touches
    (``with_structured_output(...).invoke(...)``) plus generic
    ``invoke``/``bind_tools`` so any pre-build LLM access works. The
    sub-agent itself is replaced with a stub, so this class never
    actually runs anything model-related.
    """

    base_url = "https://api.studio.nebius.ai/v1/"

    def with_structured_output(self, _schema: Any) -> "_CountingFakeLLM":
        return self

    def invoke(self, *_a: Any, **_kw: Any) -> str:
        return "structured"

    def bind_tools(self, *_a: Any, **_kw: Any) -> "_CountingFakeLLM":
        return self


def _install_fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch every ``get_llm`` import site to return the same counting fake."""

    fake = _CountingFakeLLM()

    def _factory(*_a: Any, **_kw: Any) -> _CountingFakeLLM:
        return fake

    monkeypatch.setattr(graph_mod, "get_llm", _factory)
    monkeypatch.setattr(nodes_mod, "get_llm", _factory)
    monkeypatch.setattr(tools_core_mod, "get_llm", _factory)
    monkeypatch.setattr("csa_agent.llm.get_llm", _factory)


def _route_to_react(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the router so every query is classified as STRUCTURED."""

    monkeypatch.setattr(
        graph_mod,
        "classify_query",
        lambda _query, _llm: RouteLabel.STRUCTURED,
    )


class _ScriptedSubagent:
    """Stub ReAct sub-agent that returns a canned message sequence.

    The graph's ``react_agent_node`` invokes us with
    ``state = {"messages": [profile_msg, *input_messages]}`` and then
    strips that prefix from our return value. So we must return:

    * the prefix the caller sent in (verbatim), followed by
    * the canned tool-call / tool-observation / final-answer messages
      we want the streaming wrapper to surface.
    """

    def __init__(self, canned_tail: list[BaseMessage]) -> None:
        self._canned_tail = canned_tail

    def invoke(
        self,
        state: dict[str, Any],
        _config: dict[str, Any] | None = None,
        **_kw: Any,
    ) -> dict[str, Any]:
        prefix = list(state.get("messages", []))
        return {"messages": [*prefix, *self._canned_tail]}


def _install_subagent(
    monkeypatch: pytest.MonkeyPatch, subagent: Any
) -> None:
    """Replace ``create_react_agent`` with a factory returning ``subagent``."""

    def _factory(*_a: Any, **_kw: Any) -> Any:
        return subagent

    monkeypatch.setattr(graph_mod, "create_react_agent", _factory)
    monkeypatch.setattr("langgraph.prebuilt.create_react_agent", _factory)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


_TOOL_NAME_ALPHABET = (
    "list_categories",
    "filter_by_intent",
    "filter_by_category",
    "count_rows",
    "show_examples",
    "get_intent_distribution",
)


def _tool_call_st() -> st.SearchStrategy[dict[str, Any]]:
    """Strategy for a single LangChain-style tool call dict.

    Matches the modern dict shape (``{"name", "args", "id", "type"}``)
    that ``_iter_tool_events`` reads, so the streaming helper sees a
    realistic event.
    """

    return st.builds(
        lambda name, intent_arg, n_arg, idx: {
            "name": name,
            "args": {"intent": intent_arg, "n": n_arg},
            "id": f"call_{idx}",
            "type": "tool_call",
        },
        name=st.sampled_from(_TOOL_NAME_ALPHABET),
        intent_arg=st.text(min_size=0, max_size=20),
        n_arg=st.integers(min_value=1, max_value=50),
        idx=st.integers(min_value=0, max_value=999),
    )


def _final_text_st() -> st.SearchStrategy[str]:
    """Strategy for the final-answer text.

    ``_final_answer`` requires a non-empty content string to recognise
    an AIMessage as the final answer, so we reject empty strings.
    """

    return st.text(min_size=1, max_size=40)


def _build_canned_tail(
    tool_calls: list[dict[str, Any]], final_text: str
) -> list[BaseMessage]:
    """Build the canned message sequence for :class:`_ScriptedSubagent`.

    Layout:

    * one ``AIMessage`` carrying every tool call (LangChain allows a
      single AIMessage to request multiple tools in parallel), plus
    * one ``ToolMessage`` per tool call with a matching ``tool_call_id``
      so :func:`_iter_tool_events` can pair them up, plus
    * one final ``AIMessage`` with non-empty content and no tool calls
      so :func:`_final_answer` recognises it.
    """

    # De-duplicate tool-call ids so ``ToolMessage`` pairing is
    # unambiguous. Hypothesis can occasionally generate two tool calls
    # with colliding ``idx`` values.
    seen_ids: set[str] = set()
    unique_calls: list[dict[str, Any]] = []
    for tc in tool_calls:
        if tc["id"] in seen_ids:
            continue
        seen_ids.add(tc["id"])
        unique_calls.append(tc)

    ai_with_calls = AIMessage(content="", tool_calls=unique_calls)
    tool_messages = [
        ToolMessage(content=f"observation-{i}", tool_call_id=tc["id"])
        for i, tc in enumerate(unique_calls)
    ]
    final = AIMessage(content=final_text)
    return [ai_with_calls, *tool_messages, final]


# ---------------------------------------------------------------------------
# Property 11 -- tool events strictly precede the final-answer event
# ---------------------------------------------------------------------------


@given(
    tool_calls=st.lists(_tool_call_st(), min_size=1, max_size=4),
    final_text=_final_text_st(),
)
@hsettings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_tool_events_precede_final_event(
    tool_calls: list[dict[str, Any]],
    final_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feature: customer-service-data-analyst-agent, Property 11: Tool calls are streamed before the final answer.

    Validates Requirements 4.4, 5.4, 11.3.

    For an arbitrary set of tool calls (1..4 calls) followed by a
    non-empty final-answer text, ``stream_graph`` must yield every
    ``("tool", ...)`` event strictly before the single ``("final", ...)``
    event. The final event must appear exactly once and carry the
    final-answer text we scripted.
    """

    _install_fake_llm(monkeypatch)
    canned_tail = _build_canned_tail(tool_calls, final_text)
    _install_subagent(monkeypatch, _ScriptedSubagent(canned_tail))
    _route_to_react(monkeypatch)

    graph = build_graph(checkpointer=None, df=_tiny_df())
    config = {"configurable": {"thread_id": "t-stream", "user_id": "u-stream"}}

    events = list(
        stream_graph(
            graph,
            {"messages": [HumanMessage(content="anything")]},
            config=config,
        )
    )

    # There must be at least one tool event AND exactly one final event.
    tool_indices = [i for i, e in enumerate(events) if e and e[0] == "tool"]
    final_indices = [i for i, e in enumerate(events) if e and e[0] == "final"]

    assert tool_indices, (
        "expected at least one tool event in the streamed sequence; "
        f"events={events!r}"
    )
    assert len(final_indices) == 1, (
        f"expected exactly one final event; got {len(final_indices)} in {events!r}"
    )

    final_index = final_indices[0]

    # The headline invariant of Property 11: every tool-call event index
    # is strictly less than the final-answer event index.
    for ti in tool_indices:
        assert ti < final_index, (
            f"tool event at index {ti} is not strictly before "
            f"final event at index {final_index}; events={events!r}"
        )

    # The final event must carry the scripted text so the test does
    # not silently pass on an empty fallback message.
    final_event = events[final_index]
    assert isinstance(final_event, tuple) and final_event[0] == "final"
    assert final_event[1] == final_text, (
        f"final event content {final_event[1]!r} did not match scripted "
        f"text {final_text!r}"
    )


def test_no_tool_events_when_subagent_emits_only_final_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feature: customer-service-data-analyst-agent, Property 11: Tool calls are streamed before the final answer.

    Validates Requirements 4.4, 5.4, 11.3.

    Edge case: when the run produces zero tool calls, the streaming
    helper must still terminate cleanly with a single ``("final", ...)``
    event. Property 11 is vacuously true here -- there are no tool
    events to order -- but we assert the final event still arrives so
    consumers can rely on it as a stream-completion signal.
    """

    _install_fake_llm(monkeypatch)
    final_text = "all done"
    canned_tail: list[BaseMessage] = [AIMessage(content=final_text)]
    _install_subagent(monkeypatch, _ScriptedSubagent(canned_tail))
    _route_to_react(monkeypatch)

    graph = build_graph(checkpointer=None, df=_tiny_df())
    config = {"configurable": {"thread_id": "t-stream-2", "user_id": "u-stream-2"}}

    events = list(
        stream_graph(
            graph,
            {"messages": [HumanMessage(content="anything")]},
            config=config,
        )
    )

    tool_events = [e for e in events if e and e[0] == "tool"]
    final_events = [e for e in events if e and e[0] == "final"]

    assert tool_events == []
    assert len(final_events) == 1
    assert final_events[0] == ("final", final_text)
