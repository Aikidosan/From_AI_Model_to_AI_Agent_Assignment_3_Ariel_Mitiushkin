"""Streamlit chat UI for the Customer Service Data Analyst Agent.

Run with::

    streamlit run app.py

This module implements Requirement 11 (Streamlit UI, bonus):

* Chat layout via :func:`st.chat_input` and :func:`st.chat_message`,
  with the full conversation history rendered from
  ``st.session_state.messages`` (Req 11.1).
* Sidebar inputs for *Session ID* (defaulting to a fresh ``uuid4``)
  and *User ID* (defaulting to ``"default"``) so users can resume a
  session or switch profiles (Req 11.2).
* On submit, the same compiled :func:`csa_agent.graph.build_graph` used
  by the CLI is invoked through :func:`csa_agent.graph.stream_graph`,
  and each ``("tool", name, args, observation)`` event is rendered
  inside a single :func:`st.status` block so reasoning steps appear
  *before* the final answer (Req 11.3, 11.4).

Persistence trade-off
---------------------

Per the design's "Streamlit UI" section, the CLI is the primary
persistence frontend. The Streamlit demo therefore builds the graph
*without* a checkpointer (``checkpointer=None``) to avoid plumbing the
:func:`csa_agent.checkpointer.get_checkpointer` context manager into
Streamlit's long-lived session state. Conversation history is kept
in-memory via ``st.session_state.messages`` for the duration of the
browser session; the CLI remains the path that satisfies the
durable-persistence acceptance criteria (Reqs 6.1-6.5).
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Make ``src/`` importable without an editable install.
#
# The Streamlit app lives at the repository root, while the package code
# lives under ``src/csa_agent/``. Prepending ``src/`` to ``sys.path`` lets
# users launch the app with a plain ``streamlit run app.py`` from a fresh
# checkout without first running ``pip install -e .``.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir():
    src_str = str(_SRC_DIR)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

import streamlit as st  # noqa: E402  -- import after sys.path tweak
from langchain_core.messages import HumanMessage  # noqa: E402

from csa_agent.graph import build_graph, stream_graph  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Streamlit session-state keys. Centralised so typos stay localised.
_MESSAGES_KEY = "messages"
_SESSION_ID_KEY = "session_id"
_USER_ID_KEY = "user_id"
_GRAPH_KEY = "_compiled_graph"

#: Default user id when the sidebar input is left blank.
_DEFAULT_USER_ID = "default"


# ---------------------------------------------------------------------------
# Graph caching
# ---------------------------------------------------------------------------


def _get_or_build_graph() -> Any:
    """Return a cached compiled graph, building it once per Streamlit session.

    The compiled graph is stored in ``st.session_state`` so we do not pay
    the dataset-load + tool-build cost on every interaction. The graph is
    built without a checkpointer (see module docstring for the rationale).
    """

    graph = st.session_state.get(_GRAPH_KEY)
    if graph is None:
        graph = build_graph(checkpointer=None)
        st.session_state[_GRAPH_KEY] = graph
    return graph


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _format_tool_event(name: str, args: dict[str, Any], observation: Any) -> str:
    """Render a single tool-call event as a Markdown line for ``st.status``.

    The format is intentionally compact -- one bullet per call -- so a
    multi-step reasoning trace stays readable inside the status block.
    """

    args_repr = ", ".join(f"{k}={v!r}" for k, v in (args or {}).items())
    line = f"🔧 **{name}**({args_repr})"
    if observation is not None:
        # ``observation`` may be a long string (e.g. a JSON dump from a
        # tool). Render it inside a fenced block so the formatting from
        # the tool is preserved.
        obs_text = observation if isinstance(observation, str) else repr(observation)
        line += f"\n\n```\n{obs_text}\n```"
    return line


def _render_history() -> None:
    """Render the chat history stored in ``st.session_state.messages``."""

    for entry in st.session_state[_MESSAGES_KEY]:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])


def _run_turn(prompt: str, session_id: str, user_id: str) -> str:
    """Run a single agent turn and render streaming updates.

    Tool events are written into a single :func:`st.status` block as
    they arrive; the final answer is rendered as Markdown *outside* the
    status block so it remains visible after the status collapses.
    Returns the final answer text so the caller can append it to the
    chat history.
    """

    graph = _get_or_build_graph()
    config = {
        "configurable": {
            "thread_id": session_id,
            "user_id": user_id,
        }
    }
    input_state = {"messages": [HumanMessage(content=prompt)]}

    final_content = ""
    final_placeholder = st.empty()
    with st.status("Reasoning...", expanded=True) as status:
        try:
            for event in stream_graph(graph, input_state, config=config):
                if not event:
                    continue
                kind = event[0]
                if kind == "tool":
                    _, name, args, observation = event
                    st.markdown(_format_tool_event(name, args, observation))
                elif kind == "final":
                    final_content = event[1] or ""
            status.update(label="Done", state="complete", expanded=False)
        except Exception as exc:  # noqa: BLE001 - surface failure to the UI
            status.update(label=f"Error: {exc}", state="error", expanded=True)
            raise

    final_placeholder.markdown(final_content)
    return final_content


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Render the Streamlit chat UI."""

    st.set_page_config(
        page_title="Customer Service Data Analyst Agent",
        page_icon="💬",
        layout="centered",
    )
    st.title("Customer Service Data Analyst Agent")

    # ------------------------------------------------------------------
    # One-time session-state initialisation. Doing this *before* the
    # widgets are created means we can pass ``key=`` to the inputs
    # without also passing ``value=`` (which Streamlit warns about when
    # the key already exists in session_state).
    # ------------------------------------------------------------------
    if _SESSION_ID_KEY not in st.session_state:
        st.session_state[_SESSION_ID_KEY] = str(uuid.uuid4())
    if _USER_ID_KEY not in st.session_state:
        st.session_state[_USER_ID_KEY] = _DEFAULT_USER_ID
    if _MESSAGES_KEY not in st.session_state:
        st.session_state[_MESSAGES_KEY] = []

    # ------------------------------------------------------------------
    # Sidebar: session + user inputs (Req 11.2).
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Session")
        session_id = st.text_input(
            "Session ID",
            help=(
                "Unique identifier for this conversation. Reuse a value to "
                "restore a prior session (when persistence is enabled)."
            ),
            key=_SESSION_ID_KEY,
        )
        user_id_raw = st.text_input(
            "User ID",
            help="Profile identifier used to load and update memory.",
            key=_USER_ID_KEY,
        )
        user_id = user_id_raw.strip() or _DEFAULT_USER_ID

        if st.button("Clear chat history", use_container_width=True):
            st.session_state[_MESSAGES_KEY] = []
            st.rerun()

    # ------------------------------------------------------------------
    # Main area: history + chat input (Req 11.1).
    # ------------------------------------------------------------------
    _render_history()

    prompt = st.chat_input("Ask a question about the dataset")
    if not prompt:
        return

    # Append + render the user turn first so it shows immediately.
    st.session_state[_MESSAGES_KEY].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the agent and render its turn (Req 11.3, 11.4).
    with st.chat_message("assistant"):
        final_content = _run_turn(prompt, session_id=session_id, user_id=user_id)

    st.session_state[_MESSAGES_KEY].append(
        {"role": "assistant", "content": final_content}
    )


# Streamlit executes the script top-to-bottom on every rerun, including
# under ``streamlit.testing.v1.AppTest``. Calling ``main()`` at module
# level matches the idiomatic Streamlit pattern and makes the app work
# both with ``streamlit run app.py`` and the AppTest harness used by
# task 15.2.
main()
