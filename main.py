"""Command-line entry point for the Customer Service Data Analyst Agent.

This is the CLI required by Requirement 5: a small interactive REPL that
hosts a single compiled LangGraph agent and renders each reasoning step
(tool calls + observations) before the final answer.

Usage
-----

::

    python main.py [--session SESSION_ID] [--user USER_ID] [--checkpoint-db PATH]

Behaviour
---------

* ``--session SESSION_ID`` -- optional. When omitted, a fresh ``uuid4`` is
  generated and printed so the user can copy it for a later resume
  (Requirements 5.2, 5.3, 6.2).
* ``--user USER_ID`` -- optional. Defaults to ``"default"`` so the agent
  always loads a profile (Requirements 5.6, 7.6).
* ``--checkpoint-db PATH`` -- optional. Defaults to
  ``Settings.checkpoint_db`` (``./checkpoints.db``) so persistence is on
  by default (Requirement 6.6).

Per-turn flow:

1. Read a line via ``input("> ")``.
2. ``exit`` / ``quit`` (case-insensitive, trimmed) end the loop cleanly
   (Requirement 5.5). ``KeyboardInterrupt`` prints a friendly goodbye
   and exits 0.
3. Empty lines are skipped silently.
4. Otherwise, the line is wrapped in a :class:`HumanMessage`, the graph
   is streamed via :func:`stream_graph`, and each
   ``("tool", name, args, observation)`` event is rendered as
   ``🔧 name(args) → observation`` *before* the final answer is printed
   (Requirements 4.4, 5.4).
5. Any exception raised mid-turn (notably checkpointer save failures from
   Requirement 6.5) is caught, surfaced to the user with a clear
   "(turn not acknowledged)" suffix, and the loop continues so the user
   can retry without restarting the process.

The graph is built **once** and reused across turns. The checkpointer is
opened with :func:`get_checkpointer` as a context manager, then the graph
is built and the loop runs *inside* that ``with`` block so the SQLite
connection lives for the whole session.

This module deliberately does no LLM construction itself: every LLM call
is reached through :func:`csa_agent.graph.build_graph`, which in turn
goes through :func:`csa_agent.llm.get_llm`, preserving the
"all LLM clients point at Nebius" invariant (Property 14, Requirement
9.1, 9.2).
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from typing import Any

# Make ``src/`` importable so ``import csa_agent.*`` works without an
# editable install. This mirrors how the rest of the repository expects
# to be invoked from the project root.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from langchain_core.messages import HumanMessage  # noqa: E402  (sys.path setup above)

from csa_agent.checkpointer import get_checkpointer  # noqa: E402
from csa_agent.config import get_settings  # noqa: E402
from csa_agent.dataset import get_dataset  # noqa: E402
from csa_agent.graph import build_graph, stream_graph  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Words (case-insensitive, trimmed) that exit the interactive loop.
_EXIT_WORDS: frozenset[str] = frozenset({"exit", "quit"})

#: Prefix shown next to each tool call event.
_TOOL_GLYPH: str = "🔧"

#: Prompt shown to the user at the start of every turn.
_INPUT_PROMPT: str = "> "


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the ``argparse`` parser used by :func:`main`.

    Kept as a separate function so tests can introspect the parser shape
    without invoking :func:`main`.
    """

    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "Interactive CLI for the Customer Service Data Analyst Agent. "
            "Asks questions about the Bitext customer service dataset."
        ),
    )
    parser.add_argument(
        "--session",
        dest="session",
        default=None,
        help=(
            "Session ID for conversation persistence. "
            "When omitted, a fresh UUID is generated and printed."
        ),
    )
    parser.add_argument(
        "--user",
        dest="user",
        default="default",
        help='User identifier for profile loading. Defaults to "default".',
    )
    parser.add_argument(
        "--checkpoint-db",
        dest="checkpoint_db",
        default=None,
        help=(
            "Path to the SQLite checkpoint database. "
            "Defaults to Settings.checkpoint_db (./checkpoints.db)."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _format_tool_event(name: str, args: Any, observation: Any) -> str:
    """Render a single tool event as ``🔧 name(args) → observation``.

    The contract from :func:`csa_agent.graph.stream_graph` is that ``args``
    is a ``dict`` and ``observation`` is whatever the tool returned (often
    a string already, sometimes structured). We stringify defensively so
    the CLI never crashes on an unusual observation shape.
    """

    args_repr = repr(args) if not isinstance(args, str) else args
    obs_repr = observation if isinstance(observation, str) else repr(observation)
    return f"{_TOOL_GLYPH} {name}({args_repr}) → {obs_repr}"


# ---------------------------------------------------------------------------
# Per-turn handler
# ---------------------------------------------------------------------------


def _handle_turn(
    graph: Any,
    user_query: str,
    *,
    session_id: str,
    user_id: str,
) -> None:
    """Stream one user turn and print tool events + final answer.

    Any exception raised by the graph (LLM failure, checkpointer save
    failure, tool exception bubbling through) is caught and surfaced to
    the user with a "(turn not acknowledged)" suffix per Requirement 6.5.
    The loop in :func:`main` continues so the user can retry.
    """

    config: dict[str, Any] = {
        "configurable": {"thread_id": session_id, "user_id": user_id}
    }
    input_state = {"messages": [HumanMessage(content=user_query)]}

    try:
        final_content: str | None = None
        for event in stream_graph(graph, input_state, config=config):
            if not event:
                continue
            kind = event[0]
            if kind == "tool":
                # event = ("tool", name, args, observation)
                _, name, args, observation = event
                print(_format_tool_event(name, args, observation))
            elif kind == "final":
                # event = ("final", content) -- defer printing until the
                # stream ends so any final-update tool events still beat
                # it to stdout (already guaranteed by stream_graph).
                final_content = event[1] if len(event) > 1 else ""
        if final_content:
            print(final_content)
        else:
            # ``stream_graph`` always emits a ("final", ...) event, but it
            # may carry an empty string if no AIMessage was produced.
            # Surface that explicitly so the user is not left wondering.
            print("(no answer produced)")
    except Exception as exc:  # noqa: BLE001 -- broad on purpose, see docstring
        # Requirement 6.5: on any per-turn failure (most importantly a
        # checkpointer save failure) tell the user the turn was *not*
        # acknowledged so they can retry.
        print(
            f"Error while processing turn: {exc} (turn not acknowledged)",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the interactive CLI loop and return a process exit code.

    Args:
        argv: Optional argument vector for testing. Defaults to
            ``sys.argv[1:]``.

    Returns:
        ``0`` on clean exit (``exit``/``quit``/EOF/``KeyboardInterrupt``).
    """

    args = _build_arg_parser().parse_args(argv)

    settings = get_settings()
    checkpoint_db: str = args.checkpoint_db or settings.checkpoint_db
    user_id: str = args.user or "default"

    # Generate a session id if the user didn't supply one, and surface it
    # so they can copy/reuse it later (Requirements 5.3, 6.2).
    if args.session:
        session_id = args.session
    else:
        session_id = str(uuid.uuid4())
        print(f"[Session: {session_id}]")

    # Load the dataset once up-front so any startup failures (missing
    # file, missing columns) crash before we open the checkpointer DB.
    df = get_dataset()

    # The checkpointer is a context manager; the graph and the REPL loop
    # both live inside it so the SQLite connection stays open.
    with get_checkpointer(checkpoint_db) as saver:
        graph = build_graph(df=df, checkpointer=saver)

        while True:
            try:
                line = input(_INPUT_PROMPT)
            except (KeyboardInterrupt, EOFError):
                # Friendly goodbye on Ctrl+C / Ctrl+D / piped input ending.
                print("\nGoodbye!")
                return 0

            stripped = line.strip()
            if not stripped:
                continue
            if stripped.lower() in _EXIT_WORDS:
                break

            _handle_turn(
                graph,
                stripped,
                session_id=session_id,
                user_id=user_id,
            )

    return 0


if __name__ == "__main__":  # pragma: no cover - thin shim
    raise SystemExit(main())
