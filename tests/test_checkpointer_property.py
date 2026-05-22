"""Property test for the LangGraph SQLite checkpointer factory.

Feature: customer-service-data-analyst-agent
Property 13: Checkpointer preserves message order across reopens.

Validates: Requirements 6.2, 6.4.

Strategy
--------

The :class:`SqliteSaver` ``put``/``get_tuple`` API surface has shifted
across LangGraph releases, so this test exercises it through the same
public seam the rest of the codebase uses: ``get_checkpointer(db_path)``
+ a tiny compiled graph.

The graph has a single passthrough node and a ``messages`` channel
reduced by :func:`add_messages`. Each Hypothesis example:

1. Opens the checkpointer at a fresh ``db_path``.
2. Compiles the graph with that saver.
3. For each generated message ``m``, invokes the graph with
   ``{"messages": [m]}`` so the reducer appends ``m`` to the persisted
   list (the node returns ``{}`` and contributes no messages of its
   own).
4. Closes the saver (exits the context manager), then re-opens a brand
   new saver pointing at the same file, recompiles the graph, and
   reads the persisted state via ``graph.get_state(config)``.
5. Asserts the recovered messages match the input -- in order -- on a
   ``(type_name, content)`` key. ``tool_call_id`` is intentionally
   excluded from the compare key because the reducer is allowed to
   mutate it (e.g., normalize it) and Property 13 only constrains the
   list of message identities and their order.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from hypothesis import HealthCheck, given, settings
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from csa_agent.checkpointer import get_checkpointer

from tests.strategies import message_sequences_st


# ---------------------------------------------------------------------------
# Tiny graph used purely as a vehicle for exercising the checkpointer
# ---------------------------------------------------------------------------


class _S(TypedDict, total=False):
    """Single-channel state with the standard message reducer."""

    messages: Annotated[list[BaseMessage], add_messages]


def _passthrough(_state: _S) -> dict:
    """No-op node so the graph contributes no messages of its own.

    The persisted ``messages`` list is built entirely from the
    ``{"messages": [m]}`` updates passed to ``graph.invoke``.
    """

    return {}


def _build_graph(saver):
    """Compile a one-node graph bound to ``saver``."""

    builder = StateGraph(_S)
    builder.add_node("passthrough", _passthrough)
    builder.add_edge(START, "passthrough")
    builder.add_edge("passthrough", END)
    return builder.compile(checkpointer=saver)


def _compare_key(messages: list[BaseMessage]) -> list[tuple[str, str]]:
    """Project messages onto their durable identity for ordered compare.

    We use ``type(msg).__name__`` rather than ``msg.type`` so the
    fixture is identical to the one the test docstring describes, and
    we cast ``content`` through ``str`` so the comparison is robust to
    the (rare) case where ``add_messages`` wraps the value.
    """

    return [(type(m).__name__, str(m.content)) for m in messages]


# ---------------------------------------------------------------------------
# Property 13
# ---------------------------------------------------------------------------


@given(messages=message_sequences_st())
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture,
    ],
)
def test_checkpointer_preserves_message_order_across_reopens(
    tmp_path_factory, messages: list[BaseMessage]
) -> None:
    """Feature: customer-service-data-analyst-agent, Property 13: Checkpointer preserves message order across reopens.

    Validates Requirements 6.2, 6.4.

    A sequence of messages written into a thread via the compiled
    graph survives a close/reopen cycle of the underlying SqliteSaver,
    in order, when keyed on ``(type_name, content)``.
    """

    db_path = str(tmp_path_factory.mktemp("ckpt") / "ckpt.db")
    thread_id = "t-property-13"
    config = {"configurable": {"thread_id": thread_id}}

    # ---- Phase 1: write ----------------------------------------------------
    with get_checkpointer(db_path) as saver:
        graph = _build_graph(saver)
        for msg in messages:
            graph.invoke({"messages": [msg]}, config=config)

    # ---- Phase 2: reopen and read -----------------------------------------
    with get_checkpointer(db_path) as saver:
        graph = _build_graph(saver)
        snapshot = graph.get_state(config)
        recovered = snapshot.values.get("messages", [])

    # ---- Property: ordered equality on the durable compare key ------------
    assert _compare_key(recovered) == _compare_key(messages), (
        "Recovered messages do not match the input order/content.\n"
        f"  input    = {_compare_key(messages)!r}\n"
        f"  recovered= {_compare_key(recovered)!r}"
    )
