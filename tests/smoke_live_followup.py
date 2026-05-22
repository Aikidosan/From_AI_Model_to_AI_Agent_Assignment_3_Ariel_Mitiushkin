"""Live test: conversation follow-up survives across turns and restarts.

Walks the canonical multi-turn flow from the assignment rubric:

    Turn 1: "Show me 3 examples from the REFUND category"
            -> agent shows 3 examples
    Turn 2: "Show me 3 more"
            -> agent shows 3 more from the same category
            (resolved from conversation history)

Then closes and reopens the SqliteSaver pointing at the same DB file
and asks a third question that references both prior turns to confirm
persistence across "restarts".

This script hits the real Nebius API; it is deliberately separated
from the pytest suite so a missing API key cannot fail the offline
test run. Run with::

    python tests\\smoke_live_followup.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from csa_agent.graph import build_graph, stream_graph

THREAD_ID = "smoke-live-followup-thread"
USER_ID = "smoke-live-user"


def _drain(graph, query: str) -> tuple[list, str]:
    """Run one turn end-to-end; return (tool_events, final_text)."""

    config = {"configurable": {"thread_id": THREAD_ID, "user_id": USER_ID}}
    tool_events: list = []
    final = ""
    for event in stream_graph(
        graph, {"messages": [HumanMessage(content=query)]}, config=config
    ):
        kind = event[0]
        if kind == "tool":
            _, name, args, observation = event
            tool_events.append((name, args))
            short = (
                observation[:120] + "..."
                if isinstance(observation, str) and len(observation) > 120
                else observation
            )
            print(f"  TOOL {name}({args}) -> {short!r}")
        elif kind == "final":
            final = event[1] or ""
            print(f"  FINAL: {final!r}")
    return tool_events, final


with tempfile.TemporaryDirectory() as tmp:
    db_path = os.path.join(tmp, "ckpt.db")

    # --- Session 1: turns 1 + 2 -------------------------------------------
    print("\n=== Session 1 (initial) ===")
    with SqliteSaver.from_conn_string(db_path) as saver:
        graph = build_graph(checkpointer=saver)

        print("\n--- Turn 1 ---")
        tools_1, final_1 = _drain(
            graph, "Show me 3 examples from the REFUND category."
        )
        assert tools_1, "turn 1 expected at least one tool call"
        assert any("show_examples" in name for name, _ in tools_1), (
            f"turn 1 expected show_examples, got {tools_1!r}"
        )
        assert "refund" in final_1.lower() or "REFUND" in final_1, (
            f"turn 1 final should mention refund: {final_1!r}"
        )

        print("\n--- Turn 2: 'Show me 3 more' ---")
        tools_2, final_2 = _drain(graph, "Show me 3 more.")
        # The agent must resolve "3 more" against turn 1's REFUND category.
        # show_examples should be invoked again with the same category.
        assert any(
            name == "show_examples" and args.get("category") == "REFUND"
            for name, args in tools_2
        ), (
            f"turn 2 should call show_examples(category='REFUND') based on "
            f"turn 1's history; got {tools_2!r}"
        )
        assert final_2.strip(), "turn 2 must produce a non-empty answer"

    # --- Session 2: simulate process restart ------------------------------
    print("\n=== Session 2 (post-restart) ===")
    with SqliteSaver.from_conn_string(db_path) as saver:
        graph = build_graph(checkpointer=saver)

        print("\n--- Turn 3: 'What about ORDER?' ---")
        # Rubric-style follow-up: a terse query that only makes sense
        # against the prior conversation. The agent must remember we've
        # been asking about example utterances and run the same query
        # against the ORDER category.
        tools_3, final_3 = _drain(graph, "What about ORDER?")
        # The agent should call show_examples again, this time with
        # category='ORDER', resolving the elliptical query against the
        # restored history.
        assert any(
            name == "show_examples" and args.get("category") == "ORDER"
            for name, args in tools_3
        ), (
            f"turn 3 should call show_examples(category='ORDER') based on "
            f"the restored conversation context; got {tools_3!r}"
        )
        assert final_3.strip(), "turn 3 must produce a non-empty answer"

print("\n=== PASS: conversation memory survives across turns and restarts. ===")
