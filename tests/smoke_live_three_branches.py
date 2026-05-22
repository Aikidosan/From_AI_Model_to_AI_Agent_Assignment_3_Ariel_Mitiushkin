"""Live end-to-end test: all three router branches (structured / unstructured / decline)."""
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
from csa_agent.nodes import CANONICAL_REFUSAL

CASES = [
    ("structured", "Which categories are in the dataset?"),
    ("unstructured", "Briefly describe the kinds of complaints in the FEEDBACK category."),
    ("out_of_scope", "What is the capital of France?"),
]

with tempfile.TemporaryDirectory() as tmp:
    db_path = os.path.join(tmp, "ckpt.db")
    with get_checkpointer(db_path) as saver:
        graph = build_graph(checkpointer=saver)

        for label, query in CASES:
            print(f"\n--- {label.upper()}: {query!r} ---")
            events: list[tuple] = []
            for event in stream_graph(
                graph,
                {"messages": [HumanMessage(content=query)]},
                config={
                    "configurable": {
                        "thread_id": f"smoke-{label}",
                        "user_id": "smoke-user",
                    }
                },
            ):
                events.append(event)
                if event[0] == "tool":
                    _, n, a, o = event
                    o_short = o[:100] + "..." if isinstance(o, str) and len(o) > 100 else o
                    print(f"  TOOL {n}({a}) -> {o_short!r}")
                elif event[0] == "final":
                    txt = event[1]
                    print(f"  FINAL: {txt!r}")

            tool_events = [e for e in events if e[0] == "tool"]
            final_events = [e for e in events if e[0] == "final"]
            assert final_events, f"{label}: expected final event"
            final_text = final_events[0][1]

            if label == "structured":
                assert tool_events, "structured: expected tool calls"
            elif label == "out_of_scope":
                assert not tool_events, "out_of_scope: must not call tools"
                assert final_text.strip() == CANONICAL_REFUSAL.strip(), (
                    f"out_of_scope: expected canonical refusal, got {final_text!r}"
                )

print("\nPASS: all three router branches work live.")
