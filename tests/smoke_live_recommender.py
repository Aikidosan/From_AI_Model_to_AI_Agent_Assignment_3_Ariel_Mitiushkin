"""Live test: the Query Recommender flow against real Nebius.

Walks the canonical four-turn flow from the assignment rubric:

    Turn 1: "What should I query next?"
            -> agent surfaces three suggestions, asks for confirmation
    Turn 2: "I'd rather see examples instead."
            -> agent re-rolls suggestions, still asking for confirmation
    Turn 3: "Yes, do it."
            -> agent picks the first suggestion, runs it through the
               regular pipeline, returns a real answer

This script hits the real Nebius API; it lives outside the pytest
suite so a missing API key cannot fail the offline test run. Run with::

    python tests\\smoke_live_recommender.py
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

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from csa_agent.graph import build_graph
from csa_agent.recommender import CONFIRMATION_PROMPT, SUGGESTIONS_MARKER

THREAD_ID = "smoke-live-recommender-thread"
USER_ID = "smoke-live-user"


def _user_facing_messages(state) -> list:
    """Filter to messages a user would see in the chat (skip tool calls)."""
    visible = []
    for msg in state.get("messages", []) or []:
        if isinstance(msg, AIMessage):
            if not (getattr(msg, "tool_calls", None) or []):
                visible.append(msg)
        elif isinstance(msg, HumanMessage):
            visible.append(msg)
    return visible


def _last_assistant_text(state) -> str:
    """Return the most recent assistant text, or '' if none."""
    for msg in reversed(_user_facing_messages(state)):
        if isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            return content
    return ""


with tempfile.TemporaryDirectory() as tmp:
    db_path = os.path.join(tmp, "ckpt.db")
    with SqliteSaver.from_conn_string(db_path) as saver:
        graph = build_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": THREAD_ID, "user_id": USER_ID}}

        # ---- Turn 1: trigger the recommender --------------------------
        print("\n--- Turn 1: 'What should I query next?' ---")
        state = graph.invoke(
            {"messages": [HumanMessage(content="What should I query next?")]},
            config=config,
        )
        suggestions_msg = _last_assistant_text(state)
        print(f"  ASSISTANT (truncated): {suggestions_msg[:300]!r}")
        assert SUGGESTIONS_MARKER in suggestions_msg, (
            f"turn 1 should surface suggestions; got {suggestions_msg!r}"
        )
        assert CONFIRMATION_PROMPT in suggestions_msg, (
            f"turn 1 should end with the confirmation prompt; got {suggestions_msg!r}"
        )
        assert state.get("awaiting_recommendation_confirmation") is True
        first_pending = list(state.get("pending_suggestions") or [])
        assert len(first_pending) >= 3, (
            f"recommender must produce >=3 suggestions; got {first_pending!r}"
        )

        # ---- Turn 2: refinement (free text, not a confirmation) -------
        print("\n--- Turn 2: 'I'd rather see examples instead.' ---")
        state = graph.invoke(
            {"messages": [HumanMessage(content="I'd rather see examples instead.")]},
            config=config,
        )
        refined_msg = _last_assistant_text(state)
        print(f"  ASSISTANT (truncated): {refined_msg[:300]!r}")
        assert SUGGESTIONS_MARKER in refined_msg, (
            f"turn 2 should re-surface suggestions; got {refined_msg!r}"
        )
        assert state.get("awaiting_recommendation_confirmation") is True, (
            "confirmation gate must remain armed after refinement"
        )
        refined_pending = list(state.get("pending_suggestions") or [])
        assert len(refined_pending) >= 3

        # ---- Turn 3: explicit confirmation ----------------------------
        print("\n--- Turn 3: 'Yes, do it.' ---")
        state = graph.invoke(
            {"messages": [HumanMessage(content="Yes, do it.")]},
            config=config,
        )
        final_msg = _last_assistant_text(state)
        print(f"  ASSISTANT: {final_msg[:300]!r}")
        # The gate must be cleared and a real (non-suggestion) answer
        # must have been produced via the regular pipeline.
        assert state.get("awaiting_recommendation_confirmation") in (False, None), (
            f"confirmation gate should be cleared; state="
            f"{state.get('awaiting_recommendation_confirmation')!r}"
        )
        assert SUGGESTIONS_MARKER not in final_msg, (
            f"turn 3 should produce a real answer, not another suggestion list; "
            f"got {final_msg!r}"
        )
        assert final_msg.strip(), "turn 3 must produce a non-empty final answer"

print("\n=== PASS: live recommender flow (trigger -> refine -> confirm -> execute). ===")
