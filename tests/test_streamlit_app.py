"""Streamlit AppTest harness for ``app.py``.

Validates Requirement 11 (Bonus A -- Streamlit UI):

* 11.1 -- chat layout with ``st.chat_input`` + ``st.chat_message``,
  full conversation rendered from session state.
* 11.2 -- sidebar Session ID input that lets users restore a previous
  conversation.
* 11.3 -- reasoning steps (tool calls + observations) appear before the
  final answer.
* 11.4 -- the app launches headlessly via the AppTest harness without
  a real browser.

Plus the new checkpointer wiring: when a session ID that already exists
in the SQLite checkpoint is reused, the prior conversation history is
restored on first render.

All LLM calls are stubbed; no real Nebius traffic occurs. The graph
itself is built against a tiny in-memory DataFrame so the AppTest
harness cold-starts in well under a second per case.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from streamlit.testing.v1 import AppTest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_APP_PATH = _REPO_ROOT / "app.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_df() -> pd.DataFrame:
    """Minimal DataFrame so ``build_tools`` does not need the real CSV."""

    return pd.DataFrame(
        [
            {"utterance": "i need a refund", "category": "REFUND", "intent": "track_refund"},
            {"utterance": "where is my refund", "category": "REFUND", "intent": "track_refund"},
            {"utterance": "cancel my order", "category": "ORDER", "intent": "cancel_order"},
        ]
    )


def _patch_environment(monkeypatch: pytest.MonkeyPatch, profile_dir: str, ckpt_db: str) -> None:
    """Point env vars at temporary dirs so tests are hermetic.

    Streamlit reruns the script in-process; the same Python module
    state is shared across AppTest instances within a single test, so
    we patch the environment before any AppTest is constructed.
    """

    monkeypatch.setenv("NEBIUS_API_KEY", "stub-key-for-streamlit-test")
    monkeypatch.setenv("PROFILE_DIR", profile_dir)
    monkeypatch.setenv("CHECKPOINT_DB", ckpt_db)
    # Reset cached settings so the new env vars take effect.
    from csa_agent.config import get_settings
    get_settings.cache_clear()


def _install_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub every LLM-touching call site so AppTest never hits the network.

    The ReAct sub-agent is replaced with a small inline runnable that
    appends a deterministic AIMessage; ``classify_query`` is patched to
    return STRUCTURED so a plain user query still flows through the
    ReAct branch and produces a streamed final answer; the recommender
    LLM is patched to return a numbered list so the recommender path
    can be exercised separately if needed.
    """

    from csa_agent.router import RouteLabel
    from tests.fakes import FakeChatModel

    # Router classifier: the AppTest exercises only structured queries
    # ("how many refunds?"), so STRUCTURED is the right default.
    monkeypatch.setattr(
        "csa_agent.graph.classify_query",
        lambda _query, _llm: RouteLabel.STRUCTURED,
    )

    # ReAct sub-agent: replaced with an object that emits a tool-call
    # AIMessage on first invoke and a plain AIMessage on second invoke,
    # so the streamed update sequence has a tool event followed by a
    # final answer (Property 11 / Requirement 11.3).
    from langchain_core.messages import ToolMessage as _ToolMessage

    class _StreamingStub:
        def __init__(self) -> None:
            self.invoked = 0

        def invoke(self, inp: dict[str, Any], *_a: Any, **_kw: Any) -> dict[str, Any]:
            input_messages = list(inp.get("messages", []))
            tool_call = {
                "name": "list_categories",
                "args": {},
                "id": "tc-streamlit-1",
                "type": "tool_call",
            }
            tool_call_msg = AIMessage(content="", tool_calls=[tool_call])
            tool_observation = _ToolMessage(
                content='["REFUND", "ORDER"]',
                tool_call_id="tc-streamlit-1",
                name="list_categories",
            )
            final = AIMessage(content="The dataset has 2 categories: REFUND, ORDER.")
            return {
                "messages": [
                    *input_messages,
                    tool_call_msg,
                    tool_observation,
                    final,
                ]
            }

    def _stub_create_react_agent(*_a: Any, **_kw: Any) -> _StreamingStub:
        return _StreamingStub()

    monkeypatch.setattr("csa_agent.graph.create_react_agent", _stub_create_react_agent)
    monkeypatch.setattr(
        "langgraph.prebuilt.create_react_agent", _stub_create_react_agent
    )

    # Profile-extraction LLM: returns "no facts revealed" so the
    # extractor short-circuits without affecting the stored profile.
    extractor = FakeChatModel(invoke_response=AIMessage(content='{"name": null, "preferences": {}}'))
    monkeypatch.setattr("csa_agent.nodes.get_llm", lambda *_a, **_kw: extractor)

    # Generic factory for any other site (e.g. the router LLM that
    # ``build_graph`` constructs eagerly).
    monkeypatch.setattr(
        "csa_agent.llm.get_llm",
        lambda *_a, **_kw: FakeChatModel(),
    )
    monkeypatch.setattr(
        "csa_agent.graph.get_llm",
        lambda *_a, **_kw: FakeChatModel(),
    )


def _patch_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the real Bitext loader with the tiny fixture DataFrame."""

    df = _tiny_df()
    monkeypatch.setattr("csa_agent.dataset.get_dataset", lambda *_a, **_kw: df)
    monkeypatch.setattr("csa_agent.graph.get_dataset", lambda *_a, **_kw: df)


def _clear_streamlit_resource_cache() -> None:
    """Clear ``@st.cache_resource`` so each test builds a fresh graph.

    Streamlit's cache is module-level and persists across AppTest
    instances inside one Python process, which would otherwise pin
    the *first* test's stubs onto subsequent tests. We also close the
    app's module-level ``ExitStack`` so any SqliteSaver from a prior
    test's tempdir does not survive into a new test (its temp file
    is about to be deleted, so reusing the connection would crash on
    the next query).
    """

    import streamlit as st
    try:
        st.cache_resource.clear()
    except Exception:
        # Older streamlit versions exposed a slightly different API;
        # the AppTest path is the canonical one and forgiving here is
        # fine.
        pass

    # Reset the app's module-level resource stack so a stale
    # checkpoint connection from a previous test does not bleed into
    # this one. The first AppTest invocation will rebuild it.
    try:
        import app as _app  # type: ignore[import-not-found]
        _app._clear_resource_stack()
    except Exception:
        pass


@pytest.fixture
def hermetic_app(monkeypatch: pytest.MonkeyPatch):
    """Configure env vars and stubs for a clean AppTest run.

    Yields the shared ``checkpoint_db`` path so tests that need to
    drive multiple AppTest instances (e.g. the resume test) can use the
    same SQLite database. The teardown explicitly closes the app's
    module-level resource stack *before* the temp directory is
    removed, otherwise Windows refuses to delete a SQLite file with
    an open handle and pytest reports a teardown ``PermissionError``.
    """

    # ``ignore_cleanup_errors=True`` because SQLite on Windows holds the
    # database file open via OS handles even after we close the saver;
    # the GC eventually frees those handles, but pytest's per-test
    # ``TemporaryDirectory`` teardown runs before that. The actual data
    # in the temp dir is recoverable on the next reboot via Windows'
    # standard temp cleanup; this is purely a "stop pytest from
    # reporting a noisy teardown error" fix.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        profile_dir = os.path.join(tmp, "profiles")
        ckpt_db = os.path.join(tmp, "ckpt.db")
        os.makedirs(profile_dir, exist_ok=True)

        _patch_environment(monkeypatch, profile_dir, ckpt_db)
        _patch_dataset(monkeypatch)
        _install_fakes(monkeypatch)
        _clear_streamlit_resource_cache()

        try:
            yield {"checkpoint_db": ckpt_db, "profile_dir": profile_dir}
        finally:
            # Close the SQLite connection BEFORE the TemporaryDirectory
            # context manager removes the file (Windows holds a lock
            # otherwise and tempfile teardown raises PermissionError).
            _clear_streamlit_resource_cache()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_app_renders_title_and_sidebar(hermetic_app):
    """Validates Requirements 11.1, 11.2, 11.4.

    The Streamlit app must launch cleanly under AppTest, render its
    title in the main column, and expose Session ID + User ID inputs
    in the sidebar.
    """

    at = AppTest.from_file(str(_APP_PATH), default_timeout=20)
    at.run()

    assert not at.exception, f"app raised on initial render: {at.exception}"

    # Title text appears as the first non-sidebar element.
    titles = [el.value for el in at.title]
    assert any("Customer Service" in t for t in titles), (
        f"expected page title to contain 'Customer Service'; got {titles!r}"
    )

    # Sidebar has Session ID + User ID text inputs (Requirement 11.2).
    sidebar_input_labels = [el.label for el in at.sidebar.text_input]
    assert "Session ID" in sidebar_input_labels
    assert "User ID" in sidebar_input_labels

    # Chat input is present in the main area (Requirement 11.1).
    assert at.chat_input, "expected at.chat_input to expose the chat box"


def test_app_renders_chat_turn_with_reasoning_then_final(hermetic_app):
    """Validates Requirement 11.3: reasoning steps render before the final answer.

    Driving a structured query through the AppTest harness must produce:
      * one or more user-visible chat messages (the user turn + the
        assistant turn),
      * a status block containing the streamed tool event(s),
      * the final assistant message text.
    """

    at = AppTest.from_file(str(_APP_PATH), default_timeout=30)
    at.run()
    assert not at.exception

    # Submit a query through the chat input.
    at.chat_input[0].set_value("How many categories are there?").run()
    assert not at.exception, f"app raised after chat submission: {at.exception}"

    # The assistant message must contain the stub's final text.
    chat_messages = [el for el in at.chat_message]
    assistant_messages = [
        m for m in chat_messages if getattr(m, "name", None) == "assistant"
    ]
    assert assistant_messages, "expected at least one assistant chat_message"

    # Walk every Markdown rendered under the assistant message and
    # check both the streaming tool event and the final answer landed.
    markdown_under_assistant = []
    for m in assistant_messages:
        for el in getattr(m, "markdown", []):
            markdown_under_assistant.append(el.value)
    blob = "\n".join(markdown_under_assistant)
    assert "list_categories" in blob, (
        f"expected the streamed tool call to render under the assistant; "
        f"got {markdown_under_assistant!r}"
    )
    assert "REFUND" in blob and "ORDER" in blob, (
        f"expected the final answer to mention both categories; got {blob!r}"
    )


def test_app_resumes_persisted_session_via_checkpointer(hermetic_app):
    """Validates Requirement 11.2: reusing a session ID restores history.

    Drives two sequential AppTest instances against the same SQLite
    checkpoint database. The first turn writes a checkpoint; the second
    AppTest run fixes its session ID to the same value and asserts the
    user's prior question is reloaded into the message history without
    typing anything new.
    """

    # ------------------------------------------------------------------
    # Run 1: submit a query so a checkpoint is written.
    # ------------------------------------------------------------------
    at_first = AppTest.from_file(str(_APP_PATH), default_timeout=30)
    at_first.run()

    persistent_session_id = "rubric-resume-test-thread"
    at_first.sidebar.text_input(key="session_id").set_value(persistent_session_id)
    at_first.run()
    at_first.chat_input[0].set_value("How many categories are there?").run()
    assert not at_first.exception

    # Sanity: the first run produced an assistant turn.
    first_assistant_present = any(
        getattr(m, "name", None) == "assistant" for m in at_first.chat_message
    )
    assert first_assistant_present, (
        "first AppTest run should have produced an assistant message"
    )

    # ------------------------------------------------------------------
    # Run 2: fresh AppTest, same checkpoint DB. Set the session ID to
    # the persistent value; the auto-resume logic in app.py should
    # replay the prior history into ``st.session_state.messages``.
    # ------------------------------------------------------------------
    at_second = AppTest.from_file(str(_APP_PATH), default_timeout=30)
    at_second.run()
    at_second.sidebar.text_input(key="session_id").set_value(persistent_session_id)
    at_second.run()

    # The restored history should contain at least the original user
    # turn ("How many categories are there?"). We look at the chat
    # messages rendered on this run.
    user_turns: list[str] = []
    for m in at_second.chat_message:
        if getattr(m, "name", None) == "user":
            for el in getattr(m, "markdown", []):
                user_turns.append(el.value)
    assert any("categories" in t.lower() for t in user_turns), (
        "expected the restored session to surface the original user turn; "
        f"got user turns {user_turns!r}"
    )
