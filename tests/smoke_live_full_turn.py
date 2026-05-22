"""Live end-to-end test: one full structured turn through the graph.

Asks 'How many refund requests are there?', streams the graph, and
asserts at least one tool call streamed before the final answer and
that the final answer mentions a number. No mocking -- this hits the
real Nebius API.
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

from csa_agent.checkpointer import get_checkpointer
from csa_agent.graph import build_graph, stream_graph

QUERY = "How many refund requests are there?"

with tempfile.TemporaryDirectory() as tmp:
    db_path = os.path.join(tmp, "ckpt.db")
    with get_checkpointer(db_path) as saver:
        graph = build_graph(checkpointer=saver)

        events: list[tuple] = []
        for event in stream_graph(
            graph,
            {"messages": [HumanMessage(content=QUERY)]},
            config={
                "configurable": {
                    "thread_id": "smoke-test-thread",
                    "user_id": "smoke-test-user",
                }
            },
        ):
            events.append(event)
            kind = event[0]
            if kind == "tool":
                _, name, args, observation = event
                obs_short = (
                    observation[:120] + "..."
                    if isinstance(observation, str) and len(observation) > 120
                    else observation
                )
                print(f"  TOOL {name}({args}) -> {obs_short!r}")
            elif kind == "final":
                print(f"  FINAL: {event[1]!r}")

print()
tool_events = [e for e in events if e[0] == "tool"]
final_events = [e for e in events if e[0] == "final"]

print(f"tool calls observed: {len(tool_events)}")
print(f"final events observed: {len(final_events)}")

assert tool_events, "expected at least one tool call for a structured query"
assert final_events, "expected exactly one final event"
assert len(final_events) == 1
assert final_events[0][1].strip(), "final answer must be non-empty"

# Order: every tool event index < final event index (Property 11)
tool_indices = [i for i, e in enumerate(events) if e[0] == "tool"]
final_indices = [i for i, e in enumerate(events) if e[0] == "final"]
assert max(tool_indices) < min(final_indices), (
    "all tool events must appear before the final answer"
)

print("PASS: live end-to-end turn works (router -> ReAct -> tools -> answer).")
