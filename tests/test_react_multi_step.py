"""Integration test for multi-step ReAct reasoning.

Feature: customer-service-data-analyst-agent
Task: 12.3 -- Write integration test for multi-step ReAct reasoning.

Validates: Requirements 4.1.

This test exercises the real :func:`langgraph.prebuilt.create_react_agent`
with a duck-typed fake model (:class:`tests.fakes.ScriptedReActModel`)
that emits a scripted sequence of tool calls before producing a final
answer. The fake drives the ReAct loop through two distinct tool
invocations -- ``filter_by_category`` followed by ``count_rows`` --
which the real tools (built against a small in-memory DataFrame)
execute. The streaming wrapper :func:`csa_agent.graph.stream_graph`
must surface both tool events in order, before the single
``("final", ...)`` event with the scripted answer text.

Why this complements existing tests:

* :mod:`tests.test_streaming_order` patches the entire ReAct sub-agent
  with a stub returning canned messages; it never exercises
  ``create_react_agent`` itself.
* :mod:`tests.test_react_iteration_cap` patches the sub-agent with a
  ``GraphRecursionError``-raising stub; again, no real ReAct loop runs.
* This test patches only the LLM (so no Nebius traffic) and lets the
  real ReAct loop coordinate tool calls -- the multi-step contract from
  Requirement 4.1.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from langchain_core.messages import HumanMessage

import csa_agent.graph as graph_mod
import csa_agent.nodes as nodes_mod
import csa_agent.tools.core as tools_core_mod
from csa_agent.graph import build_graph, stream_graph
from csa_agent.router import RouteLabel
from tests.fakes import ScriptedReActModel


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _two_category_df() -> pd.DataFrame:
    """Return a small DataFrame with two categories and a few rows each.

    The fake model's final answer claims there are *three* refund
    requests, so the REFUND slice carries exactly three rows. The ORDER
    rows ensure the dataset has more than one category, so
    ``filter_by_category`` is doing real filtering work.
    """

    return pd.DataFrame(
        [
            {
                "utterance": "i want a refund",
                "category": "REFUND",
                "intent": "track_refund",
            },
            {
                "utterance": "where is my refund",
                "category": "REFUND",
                "intent": "track_refund",
            },
            {
                "utterance": "refund please",
                "category": "REFUND",
                "intent": "get_refund",
            },
            {
                "utterance": "where is my order",
                "category": "ORDER",
                "intent": "track_order",
            },
            {
                "utterance": "cancel my order",
                "category": "ORDER",
                "intent": "cancel_order",
            },
        ]
    )


def _install_fake_llm(monkeypatch: pytest.MonkeyPatch, model: Any) -> None:
    """Patch every ``get_llm`` import site to return the same fake model.

    The graph, nodes, and tools.core modules each import ``get_llm`` at
    module load time, so patching the source module alone is not
    enough -- every binding must be replaced.
    """

    def _factory(*_a: Any, **_kw: Any) -> Any:
        return model

    monkeypatch.setattr(graph_mod, "get_llm", _factory)
    monkeypatch.setattr(nodes_mod, "get_llm", _factory)
    monkeypatch.setattr(tools_core_mod, "get_llm", _factory)
    monkeypatch.setattr("csa_agent.llm.get_llm", _factory)


def _route_to_react(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the router so every query is classified as STRUCTURED.

    Avoids an outbound classifier call and routes the run into the
    ReAct branch where the multi-step reasoning happens.
    """

    monkeypatch.setattr(
        graph_mod,
        "classify_query",
        lambda _query, _llm: RouteLabel.STRUCTURED,
    )


# ---------------------------------------------------------------------------
# Requirement 4.1 -- multi-step ReAct reasoning
# ---------------------------------------------------------------------------


def test_react_multi_step_streams_tool_calls_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feature: customer-service-data-analyst-agent. Validates Requirement 4.1.

    With a scripted fake model that emits ``filter_by_category`` then
    ``count_rows`` before answering, ``stream_graph`` must yield both
    tool events in scripted order, both strictly before the single
    ``("final", ...)`` event carrying the scripted answer text.
    """

    final_text = "There are 3 refund requests in the dataset."
    fake_model = ScriptedReActModel(
        scripted=[
            ("filter_by_category", {"category": "REFUND"}),
            ("count_rows", {"category": "REFUND"}),
        ],
        final_text=final_text,
    )

    _install_fake_llm(monkeypatch, fake_model)
    _route_to_react(monkeypatch)

    graph = build_graph(checkpointer=None, df=_two_category_df())
    config = {
        "configurable": {
            "thread_id": "t-react-multi",
            "user_id": "u-react-multi",
        }
    }

    events = list(
        stream_graph(
            graph,
            {"messages": [HumanMessage(content="How many refund requests are there?")]},
            config=config,
        )
    )

    tool_events = [
        (idx, e) for idx, e in enumerate(events) if e and e[0] == "tool"
    ]
    final_events = [
        (idx, e) for idx, e in enumerate(events) if e and e[0] == "final"
    ]

    # Exactly two tool events must appear, in scripted order.
    assert len(tool_events) == 2, (
        f"expected exactly 2 tool events, got {len(tool_events)}; "
        f"events={events!r}"
    )

    first_idx, first_event = tool_events[0]
    second_idx, second_event = tool_events[1]
    assert first_event[1] == "filter_by_category", (
        f"first tool event was not filter_by_category; got {first_event!r}"
    )
    assert second_event[1] == "count_rows", (
        f"second tool event was not count_rows; got {second_event!r}"
    )

    # Exactly one final event must appear, after both tool events.
    assert len(final_events) == 1, (
        f"expected exactly one final event; got {len(final_events)} "
        f"in {events!r}"
    )
    final_idx, final_event = final_events[0]
    assert first_idx < final_idx, (
        f"filter_by_category event at index {first_idx} is not before "
        f"final event at index {final_idx}; events={events!r}"
    )
    assert second_idx < final_idx, (
        f"count_rows event at index {second_idx} is not before "
        f"final event at index {final_idx}; events={events!r}"
    )

    # The final answer must carry the scripted text.
    assert final_text in final_event[1], (
        f"final event content {final_event[1]!r} does not contain "
        f"scripted text {final_text!r}"
    )
