"""End-to-end integration tests for the Query Recommender feature (Bonus B).

Validates the full four-turn flow described in the assignment:

    1. User: "What should I query next?"
       Agent: <three suggestions, asks for confirmation>
    2. User: "I'd rather see examples instead."
       Agent: <revised suggestions, asks again>
    3. User: "Yes, do it."
       Agent: <selected suggestion is executed via the regular pipeline>

All LLM calls are stubbed via :class:`tests.fakes.FakeChatModel` /
:class:`ScriptedReActModel` so this file does not hit Nebius. The
test exercises the *real* compiled graph end-to-end so it catches
wiring regressions in :func:`csa_agent.graph.build_graph`.

Validates Requirements 12.1, 12.2, 12.3, 12.4, 12.5 and Property 16.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Iterator

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from csa_agent.graph import build_graph
from csa_agent.recommender import CONFIRMATION_PROMPT, SUGGESTIONS_MARKER

from tests.fakes import FakeChatModel, ScriptedReActModel, llm_factory_returning


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def small_df() -> pd.DataFrame:
    """A tiny but realistic dataset so the ReAct path has data to act on."""

    rows = [
        {"utterance": f"refund question {i}", "category": "REFUND", "intent": "track_refund"}
        for i in range(5)
    ] + [
        {"utterance": f"cancel question {i}", "category": "ORDER", "intent": "cancel_order"}
        for i in range(3)
    ]
    return pd.DataFrame(rows, columns=["utterance", "category", "intent"])


@pytest.fixture
def stub_llms(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch every ``get_llm`` call site so no Nebius traffic occurs.

    Three call sites are patched:

    * ``csa_agent.llm.get_llm`` (the central factory).
    * ``csa_agent.tools.core.get_llm`` (used by ``summarize_category``).
    * ``csa_agent.recommender.get_llm`` (used by the recommender).
    * ``csa_agent.graph.get_llm`` (the router classifier and the ReAct
      sub-agent).

    The fakes return tool-call-emitting messages for the ReAct path and
    a fixed numbered-list response for the recommender so suggestion
    generation does not touch the network.
    """

    # The recommender's LLM returns a numbered list. We hand-craft the
    # response so the parser produces exactly three suggestions and the
    # test can assert on their text without coupling to LLM creativity.
    recommender_response = AIMessage(
        content=(
            "1. Show me 5 examples from the REFUND category\n"
            "2. List all categories\n"
            "3. What is the distribution of intents in REFUND?"
        )
    )
    recommender_llm = FakeChatModel(invoke_response=recommender_response)

    # The ReAct agent's LLM emits one tool call (filter_by_category)
    # then a final answer. Sufficient to verify the confirmation path
    # routes through the regular ReAct pipeline.
    react_llm = ScriptedReActModel(
        scripted=[("filter_by_category", {"category": "REFUND"})],
        final_text="Here are 5 refund examples.",
    )

    # The router LLM uses ``with_structured_output``; we configure the
    # FakeChatModel to return the STRUCTURED label so the injected
    # confirmed query takes the ReAct path.
    from csa_agent.router import RouteLabel, _RouterDecision

    router_llm = FakeChatModel(
        structured_payload=_RouterDecision(label=RouteLabel.STRUCTURED)
    )

    # Patch every site that imports get_llm. We can't share a single
    # FakeChatModel because the recommender, router, and ReAct agent
    # all exercise different surfaces of the LLM API.
    monkeypatch.setattr("csa_agent.llm.get_llm", llm_factory_returning(router_llm))
    monkeypatch.setattr(
        "csa_agent.recommender.get_llm",
        llm_factory_returning(recommender_llm),
    )
    monkeypatch.setattr(
        "csa_agent.graph.get_llm",
        # The graph constructs *two* LLM clients on build: the router
        # classifier and the ReAct model. Return the appropriate fake
        # by counting calls -- first call is the router, second the
        # ReAct agent.
        _alternating_factory(router_llm, react_llm),
    )

    return {
        "recommender_llm": recommender_llm,
        "react_llm": react_llm,
        "router_llm": router_llm,
    }


def _alternating_factory(*models: Any):
    """Return a ``get_llm``-style callable that hands out ``models`` in order.

    After the supplied models are exhausted, the last one is returned for
    every subsequent call. Used so :func:`build_graph` can construct a
    classifier LLM (call 1) and a ReAct LLM (call 2) without sharing the
    same FakeChatModel surface.
    """

    state = {"i": 0}

    def _factory(*_a: Any, **_kw: Any) -> Any:
        i = min(state["i"], len(models) - 1)
        state["i"] += 1
        return models[i]

    return _factory


def _user_visible_messages(state: dict[str, Any]) -> list[Any]:
    """Filter ``state['messages']`` down to the messages a user would see.

    Skips ``ToolMessage`` entries and AIMessages that carry only tool
    calls, leaving the conversation exactly as the CLI would render it.
    """

    out: list[Any] = []
    for msg in state.get("messages", []) or []:
        if isinstance(msg, AIMessage):
            if not (getattr(msg, "tool_calls", None) or []):
                out.append(msg)
        elif isinstance(msg, HumanMessage):
            out.append(msg)
    return out


@pytest.fixture
def stateful_graph(small_df, stub_llms) -> Iterator[Any]:
    """Yield a compiled graph backed by a real :class:`SqliteSaver`.

    Multi-turn integration tests need a checkpointer so prior-turn state
    (notably ``awaiting_recommendation_confirmation`` and
    ``pending_suggestions``) survives between :func:`graph.invoke` calls.
    Without one, each invoke starts from scratch and the confirmation
    gate can never observe its own armed state.
    """

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "ckpt.db")
        with SqliteSaver.from_conn_string(db_path) as saver:
            graph = build_graph(df=small_df, checkpointer=saver)
            yield graph


# ---------------------------------------------------------------------------
# End-to-end recommender flow
# ---------------------------------------------------------------------------


def test_recommender_trigger_surfaces_suggestions(stateful_graph, stub_llms):
    """Turn 1: trigger phrase produces a suggestion list with confirmation prompt.

    Validates Requirements 12.1, 12.5 and Property 16.
    """

    config = {"configurable": {"thread_id": "rec-1", "user_id": "alice"}}

    state = stateful_graph.invoke(
        {"messages": [HumanMessage(content="What should I query next?")]},
        config=config,
    )

    # The agent must produce a suggestion message, must NOT execute any
    # downstream query, and must arm the confirmation gate so the next
    # turn lands in confirmation_node.
    visible = _user_visible_messages(state)
    assert visible, "expected at least one user-visible message"
    final = visible[-1]
    assert isinstance(final, AIMessage)
    assert SUGGESTIONS_MARKER in final.content
    assert CONFIRMATION_PROMPT in final.content

    # No ReAct tool calls happened (Property 16).
    assert stub_llms["react_llm"].invocations == [], (
        "recommender turn must not invoke the ReAct sub-agent"
    )
    # Confirmation gate is armed.
    assert state.get("awaiting_recommendation_confirmation") is True
    pending = state.get("pending_suggestions") or []
    assert len(pending) >= 3


def test_recommender_full_flow_refine_then_confirm(stateful_graph, stub_llms):
    """Walks the rubric's example: trigger -> refine -> confirm -> execute.

    Validates Requirements 12.1, 12.2, 12.3, 12.4, 12.5.
    """

    config = {"configurable": {"thread_id": "rec-2", "user_id": "alice"}}

    # Turn 1: trigger.
    state = stateful_graph.invoke(
        {"messages": [HumanMessage(content="What should I query next?")]},
        config=config,
    )
    assert state.get("awaiting_recommendation_confirmation") is True

    # Turn 2: refinement (free-text reply that is neither a number nor a
    # confirmation phrase). The agent should re-roll suggestions and
    # keep the gate armed.
    state = stateful_graph.invoke(
        {"messages": [HumanMessage(content="I'd rather see examples instead.")]},
        config=config,
    )
    assert state.get("awaiting_recommendation_confirmation") is True
    visible = _user_visible_messages(state)
    refined_msg = visible[-1]
    assert isinstance(refined_msg, AIMessage)
    assert SUGGESTIONS_MARKER in refined_msg.content
    # Still no downstream execution.
    assert stub_llms["react_llm"].invocations == [], (
        "refinement turn must not invoke the ReAct sub-agent"
    )

    # Turn 3: explicit confirmation. The agent should pick the first
    # pending suggestion, route it through the regular pipeline, and
    # produce a real answer.
    state = stateful_graph.invoke(
        {"messages": [HumanMessage(content="Yes, do it.")]},
        config=config,
    )
    assert state.get("awaiting_recommendation_confirmation") is False
    assert state.get("pending_suggestions") in (None, [])
    # The injected suggestion went through ReAct -> at least one call
    # to the ReAct LLM.
    assert stub_llms["react_llm"].invocations, (
        "confirmation turn must invoke the ReAct sub-agent"
    )
    visible = _user_visible_messages(state)
    final = visible[-1]
    assert isinstance(final, AIMessage)
    # The scripted ReAct agent emits "Here are 5 refund examples." as
    # its terminal message.
    assert "refund examples" in final.content.lower()


def test_recommender_numeric_pick_executes_chosen_suggestion(stateful_graph, stub_llms):
    """Picking a number (e.g. "2") confirms that specific suggestion.

    Validates Requirement 12.3.
    """

    config = {"configurable": {"thread_id": "rec-3", "user_id": "alice"}}

    state = stateful_graph.invoke(
        {"messages": [HumanMessage(content="What should I query next?")]},
        config=config,
    )
    pending_before = list(state.get("pending_suggestions") or [])
    assert len(pending_before) >= 3

    state = stateful_graph.invoke(
        {"messages": [HumanMessage(content="2")]},
        config=config,
    )

    # The injected query should be the *second* pending suggestion. We
    # find it by walking the message list backwards looking for the
    # most recent HumanMessage that does NOT match the user's literal
    # reply ("2"); that one was injected by confirmation_node.
    injected = None
    for msg in reversed(state.get("messages", []) or []):
        if isinstance(msg, HumanMessage) and msg.content != "2":
            injected = msg.content
            break
    assert injected is not None, "no injected query found in message history"
    assert injected == pending_before[1], (
        f"expected the injected query to equal pending[1]={pending_before[1]!r}, "
        f"got {injected!r}"
    )


def test_recommender_rejection_clears_state_without_executing(stateful_graph, stub_llms):
    """A "no" reply ends the turn cleanly with no downstream query.

    Validates Requirement 12.5.
    """

    config = {"configurable": {"thread_id": "rec-4", "user_id": "alice"}}

    stateful_graph.invoke(
        {"messages": [HumanMessage(content="What should I query next?")]},
        config=config,
    )
    state = stateful_graph.invoke(
        {"messages": [HumanMessage(content="no")]},
        config=config,
    )

    assert state.get("awaiting_recommendation_confirmation") is False
    assert state.get("pending_suggestions") in (None, [])
    assert stub_llms["react_llm"].invocations == [], (
        "rejection must not invoke the ReAct sub-agent"
    )
    visible = _user_visible_messages(state)
    final = visible[-1]
    assert isinstance(final, AIMessage)
    assert "no problem" in final.content.lower()
