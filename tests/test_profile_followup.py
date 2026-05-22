"""Integration test for the "what do you remember about me?" flow.

Feature: customer-service-data-analyst-agent
Task 12.4

Validates: Requirements 7.3.

When a user has an existing :class:`csa_agent.profile.UserProfile`
saved on disk and asks the agent something like "what do you remember
about me?", the agent's final answer must surface the stored profile
data (name, frequent topics, preferences). This is the user-visible
side of Requirement 7.3 -- "the agent uses the stored profile to
personalize subsequent answers" -- and the integration point this
test pins down is the system-message injection inside
:func:`csa_agent.graph._make_react_agent_node`.

Test design
-----------

The agent's ReAct branch is normally backed by an LLM. Calling Nebius
from the test suite would be slow, flaky, and dependent on a live API
key, so we substitute a stub sub-agent whose ``invoke`` inspects the
messages it receives, finds the :class:`SystemMessage` carrying the
profile context (the one ``react_agent_node`` prepends per
Requirement 7.3), and echoes that text back as a final
:class:`AIMessage`. This proves end-to-end that:

* :func:`csa_agent.nodes.load_user_profile_node` reads the on-disk
  profile keyed by ``config["configurable"]["user_id"]`` (Requirement 7.6).
* The stored ``UserProfile`` is threaded into the graph state and
  reaches the ReAct branch.
* :func:`csa_agent.graph._profile_context_message` materialises the
  name / frequent topics / preferences into the SystemMessage
  prepended to the sub-agent's input.

If any link in that chain regresses, the echoed final answer will
not contain the stored profile fields and the assertions below will
fail.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

import csa_agent.graph as graph_mod
import csa_agent.nodes as nodes_mod
import csa_agent.tools.core as tools_core_mod
from csa_agent.graph import build_graph, stream_graph
from csa_agent.profile import UserProfile, save_profile
from csa_agent.router import RouteLabel


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _tiny_df() -> pd.DataFrame:
    """Return a minimal DataFrame so build_tools succeeds without the real CSV."""

    return pd.DataFrame(
        [
            {
                "utterance": "I want a refund",
                "category": "REFUND",
                "intent": "track_refund",
            },
            {
                "utterance": "cancel my order",
                "category": "ORDER",
                "intent": "cancel_order",
            },
        ]
    )


class _CountingFakeLLM:
    """Tiny stand-in for ChatOpenAI used by the router and tool factories.

    Exposes the surface :func:`csa_agent.router.classify_query` touches
    (``with_structured_output(...).invoke(...)``) plus the generic
    ``invoke``/``bind_tools`` methods other code paths may call. The
    router is patched outright in this test, so the structured-output
    path is not actually exercised, but we still implement it so a
    stray import-time access cannot blow up.
    """

    base_url = "https://api.studio.nebius.ai/v1/"

    def with_structured_output(self, _schema: Any) -> "_CountingFakeLLM":
        return self

    def invoke(self, *_a: Any, **_kw: Any) -> str:
        return "structured"

    def bind_tools(self, *_a: Any, **_kw: Any) -> "_CountingFakeLLM":
        return self


def _install_fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch every ``get_llm`` import site to return the same counting fake.

    The ``csa_agent.graph``, ``csa_agent.nodes``, and
    ``csa_agent.tools.core`` modules each capture ``get_llm`` at import
    time, so patching the source module alone is not sufficient -- we
    must overwrite every binding that holds a reference.
    """

    fake = _CountingFakeLLM()

    def _factory(*_a: Any, **_kw: Any) -> _CountingFakeLLM:
        return fake

    monkeypatch.setattr(graph_mod, "get_llm", _factory)
    monkeypatch.setattr(nodes_mod, "get_llm", _factory)
    monkeypatch.setattr(tools_core_mod, "get_llm", _factory)
    monkeypatch.setattr("csa_agent.llm.get_llm", _factory)


class _EchoProfileSubagent:
    """Stub ReAct sub-agent that echoes the injected profile SystemMessage.

    The graph's ``react_agent_node`` invokes us with::

        state = {"messages": [profile_system_msg, *original_messages]}

    where ``profile_system_msg`` is the :class:`SystemMessage` produced
    by :func:`csa_agent.graph._profile_context_message` from the
    user's stored profile. To prove that message reaches the
    sub-agent, we walk the input messages, collect the content of any
    SystemMessage that looks like the profile context block (the
    template begins with "User profile context:"), and return it as
    a final AIMessage.

    The graph then strips the prefix it sent in (``1 + len(messages)``
    items) and keeps only the trailing AIMessage, which becomes the
    user-visible final answer.
    """

    def __init__(self) -> None:
        self.invocations: list[list[BaseMessage]] = []

    def invoke(
        self,
        state: dict[str, Any],
        _config: dict[str, Any] | None = None,
        **_kw: Any,
    ) -> dict[str, Any]:
        messages: list[BaseMessage] = list(state.get("messages", []))
        self.invocations.append(messages)

        # Collect every SystemMessage whose content looks like the
        # profile-context block. Using a substring check rather than
        # ``isinstance`` alone keeps the test robust if other
        # SystemMessages are ever prepended in the future.
        profile_chunks: list[str] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                content = msg.content
                if isinstance(content, str) and "User profile" in content:
                    profile_chunks.append(content)

        echoed = "\n".join(profile_chunks) if profile_chunks else "no profile found"
        # Echo the profile context back to the user verbatim so the
        # graph's downstream wrapper has a non-empty AIMessage to
        # surface as the final answer.
        return {"messages": [*messages, AIMessage(content=echoed)]}


def _install_subagent(
    monkeypatch: pytest.MonkeyPatch, subagent: Any
) -> None:
    """Replace ``create_react_agent`` with a factory returning ``subagent``.

    The real :func:`langgraph.prebuilt.create_react_agent` builds a
    LangChain runnable (``prompt | model``) which rejects duck-typed
    fake LLMs. Patching the constructor itself bypasses that path
    while preserving the ``invoke`` shape the parent graph expects.
    """

    def _factory(*_a: Any, **_kw: Any) -> Any:
        return subagent

    monkeypatch.setattr(graph_mod, "create_react_agent", _factory)
    # ``nodes._build_summarize_subagent`` performs an in-function
    # import of ``langgraph.prebuilt.create_react_agent``; patch the
    # canonical location too so neither code path constructs a real
    # ReAct agent.
    monkeypatch.setattr("langgraph.prebuilt.create_react_agent", _factory)


def _route_to_react(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the router so every query is classified as STRUCTURED.

    The "what do you remember about me?" query would normally trip
    the router into one of the three labels depending on the
    classifier LLM's behaviour; we force STRUCTURED so the test
    exercises the ReAct branch where profile context is injected.
    """

    monkeypatch.setattr(
        graph_mod,
        "classify_query",
        lambda _query, _llm: RouteLabel.STRUCTURED,
    )


def _override_profile_dir(
    monkeypatch: pytest.MonkeyPatch, profile_dir: Path
) -> None:
    """Point the cached :class:`Settings` at ``profile_dir`` for this test.

    Setting ``PROFILE_DIR`` via ``monkeypatch.setenv`` alone is not
    enough because :func:`csa_agent.config.get_settings` is cached
    by ``functools.lru_cache``. The autouse fixture in
    ``conftest.py`` clears that cache between tests, but we re-clear
    here to be defensive against ordering surprises.
    """

    monkeypatch.setenv("PROFILE_DIR", str(profile_dir))
    from csa_agent.config import get_settings as _get_settings

    _get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_what_do_you_remember_about_me_surfaces_stored_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Feature: customer-service-data-analyst-agent, Task 12.4.

    Validates Requirement 7.3.

    Pre-populates a :class:`UserProfile` for ``user_id="alice"`` on
    disk, runs the graph against the question "what do you remember
    about me?", and asserts the agent's final answer surfaces the
    stored name, frequent topic, and preference. The ReAct sub-agent
    is replaced with a stub that echoes the profile-context
    SystemMessage prepended by ``react_agent_node`` so the test
    exercises the real profile-load and message-injection wiring
    without invoking Nebius.
    """

    # 1. Point Settings.profile_dir at a fresh tmp directory so
    #    load_profile / save_profile both target the same location.
    _override_profile_dir(monkeypatch, tmp_path)

    # 2. Pre-populate the profile on disk. ``save_profile`` accepts an
    #    explicit ``profile_dir`` override to keep this independent of
    #    the env-var path resolution above, which we rely on for
    #    ``load_user_profile_node`` (which has no direct override).
    profile = UserProfile(
        user_id="alice",
        name="Alice",
        frequent_topics=["REFUND", "ORDER"],
        preferences={"language": "english"},
    )
    saved = save_profile(profile, profile_dir=str(tmp_path))
    assert (tmp_path / "alice.json").is_file(), (
        f"pre-populated profile was not written to disk; "
        f"directory contents: {list(tmp_path.iterdir())}"
    )
    # Sanity: saved object reflects what we wrote so a future
    # regression in save_profile cannot pass this test silently.
    assert saved.name == "Alice"
    assert "REFUND" in saved.frequent_topics
    assert saved.preferences.get("language") == "english"

    # 3. Wire the test doubles. Order matters: install LLM and
    #    sub-agent fakes *before* build_graph so eager construction
    #    of the router and ReAct sub-agent never reach Nebius.
    _install_fake_llm(monkeypatch)
    stub = _EchoProfileSubagent()
    _install_subagent(monkeypatch, stub)
    _route_to_react(monkeypatch)

    # 4. Build the graph against the in-memory DataFrame so the tool
    #    factory does not load the real Bitext CSV.
    graph = build_graph(checkpointer=None, df=_tiny_df())

    # 5. Stream a single turn for user_id="alice". Collecting the
    #    streamed events lets us pull out the canonical
    #    ``("final", <text>)`` event the streaming wrapper guarantees.
    config = {
        "configurable": {"thread_id": "t-profile-followup", "user_id": "alice"},
    }
    events = list(
        stream_graph(
            graph,
            {"messages": [HumanMessage(content="what do you remember about me?")]},
            config=config,
        )
    )

    # The stub must have been invoked at least once -- otherwise we
    # never routed to the ReAct branch and the assertions below would
    # be testing the wrong thing.
    assert stub.invocations, (
        "echo sub-agent was never invoked; routing did not reach "
        "react_agent_node"
    )

    # 6. Locate the single ``("final", ...)`` event and pull out its
    #    text. ``stream_graph`` always emits exactly one final event.
    final_events = [e for e in events if e and e[0] == "final"]
    assert len(final_events) == 1, (
        f"expected exactly one final event from stream_graph; "
        f"got {len(final_events)} in {events!r}"
    )
    final_text = final_events[0][1]
    assert isinstance(final_text, str) and final_text.strip(), (
        f"final event content is empty: {final_text!r}"
    )

    # 7. Headline assertions: the final answer must surface every
    #    field of the stored profile. Using ``in`` rather than exact
    #    equality keeps the test robust to template tweaks in
    #    ``_profile_context_message`` (e.g. reordering the bullets,
    #    adding a header) provided the substantive data still appears.
    assert "Alice" in final_text, (
        f"final answer did not mention the stored name 'Alice'; "
        f"got {final_text!r}"
    )
    assert "REFUND" in final_text, (
        f"final answer did not mention the stored frequent topic 'REFUND'; "
        f"got {final_text!r}"
    )
    assert "english" in final_text, (
        f"final answer did not mention the stored preference 'english'; "
        f"got {final_text!r}"
    )
