"""Tests for the natural-conversation profile fact extractor (Task 2b).

The extractor is the small Nebius call invoked by
:func:`csa_agent.nodes.update_profile_node` after every turn. It reads
the user's most recent message and returns either an empty dict (no
durable facts revealed) or a dict with optional ``name`` / ``preferences``
keys. The agent merges that into the per-user profile so future
"what do you remember about me?" turns see the extracted information.

These tests exercise the extractor in isolation (via a
:class:`tests.fakes.FakeChatModel`) and the full integration with
:func:`update_profile_node`, asserting:

* Names and preferences are merged into the persisted profile.
* Empty / null payloads leave the profile unchanged.
* Unknown preference keys are dropped (whitelist enforcement).
* A malformed JSON response from the LLM is silently ignored.
* Repeated turns never double-set the name (idempotence on existing
  values).

Validates: Requirements 7.1, 7.2, 7.5.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from csa_agent.nodes import (
    _ALLOWED_PREFERENCE_KEYS,
    _extract_profile_facts_from_turn,
    update_profile_node,
)
from csa_agent.profile import UserProfile

from tests.fakes import FakeChatModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm_returning(content: str) -> FakeChatModel:
    """Build a ``FakeChatModel`` whose ``invoke`` returns a fixed AIMessage."""

    return FakeChatModel(invoke_response=AIMessage(content=content))


# ---------------------------------------------------------------------------
# Pure-extractor tests
# ---------------------------------------------------------------------------


def test_extractor_pulls_name_from_introduction():
    """A plain-JSON response with a name yields a populated dict."""

    llm = _llm_returning('{"name": "Ariel", "preferences": {}}')
    out = _extract_profile_facts_from_turn("My name is Ariel.", llm=llm)

    assert out == {"name": "Ariel"}
    assert llm.invocations, "extractor must call the LLM"


def test_extractor_pulls_preferences_with_whitelisted_keys():
    """Recognised preference keys (tone, verbosity, etc.) survive normalisation."""

    llm = _llm_returning(
        '{"name": null, "preferences": {"tone": "brief", "verbosity": "concise"}}'
    )
    out = _extract_profile_facts_from_turn("Keep answers short, please.", llm=llm)

    assert "name" not in out
    assert out["preferences"] == {"tone": "brief", "verbosity": "concise"}


def test_extractor_drops_unknown_preference_keys():
    """Anything outside the whitelist (e.g. random keys) is filtered out."""

    llm = _llm_returning(
        '{"preferences": {"foo": "bar", "tone": "warm", "drop_me": "x"}}'
    )
    out = _extract_profile_facts_from_turn("anything", llm=llm)

    # Only "tone" survives the whitelist filter.
    assert out["preferences"] == {"tone": "warm"}
    for key in out["preferences"]:
        assert key in _ALLOWED_PREFERENCE_KEYS


def test_extractor_handles_markdown_fenced_json():
    """The model occasionally wraps JSON in ```...``` fences; strip them."""

    fenced = '```json\n{"name": "Sasha", "preferences": {}}\n```'
    llm = _llm_returning(fenced)
    out = _extract_profile_facts_from_turn("hi I'm Sasha", llm=llm)

    assert out == {"name": "Sasha"}


def test_extractor_returns_empty_on_malformed_json():
    """A non-JSON response is silently ignored (returns empty dict)."""

    llm = _llm_returning("not json at all")
    out = _extract_profile_facts_from_turn("anything", llm=llm)

    assert out == {}


def test_extractor_returns_empty_when_llm_raises():
    """Network or auth failures must never bubble up to the caller."""

    llm = FakeChatModel(raise_on_call=RuntimeError)
    out = _extract_profile_facts_from_turn("anything", llm=llm)

    assert out == {}


def test_extractor_returns_empty_for_empty_input():
    """A blank user message short-circuits without calling the LLM."""

    llm = _llm_returning('{"name": "x"}')
    out = _extract_profile_facts_from_turn("   ", llm=llm)

    assert out == {}
    assert llm.invocations == [], "no LLM call expected for empty input"


def test_extractor_truncates_overlong_input():
    """Inputs longer than the cap are truncated before being sent to the LLM."""

    long_text = "x" * 5000
    llm = _llm_returning('{"name": null}')
    _extract_profile_facts_from_turn(long_text, llm=llm)

    sent = llm.invocations[0]
    # The fake captures the raw messages list passed to invoke. The
    # human content is the second tuple's second element.
    human_content = sent[1][1]
    assert len(human_content) <= 600


# ---------------------------------------------------------------------------
# Integration with update_profile_node
# ---------------------------------------------------------------------------


def test_update_profile_node_persists_extracted_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """Driving an introduction through update_profile_node persists the name."""

    monkeypatch.setenv("PROFILE_DIR", str(tmp_path))

    fake = _llm_returning('{"name": "Ariel", "preferences": {}}')
    monkeypatch.setattr("csa_agent.nodes.get_llm", lambda *_a, **_kw: fake)

    profile = UserProfile(user_id="alice")
    state: dict[str, Any] = {
        "user_profile": profile,
        "messages": [HumanMessage(content="Hi, my name is Ariel.")],
    }
    config = {"configurable": {"user_id": "alice"}}

    update = update_profile_node(state, config=config)

    saved = update["user_profile"]
    assert saved.name == "Ariel"


def test_update_profile_node_persists_extracted_preferences(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """Preferences mentioned in conversation are merged into the profile."""

    monkeypatch.setenv("PROFILE_DIR", str(tmp_path))
    fake = _llm_returning(
        '{"name": null, "preferences": {"tone": "brief", "format": "bullets"}}'
    )
    monkeypatch.setattr("csa_agent.nodes.get_llm", lambda *_a, **_kw: fake)

    profile = UserProfile(user_id="bob", preferences={"language": "en"})
    state: dict[str, Any] = {
        "user_profile": profile,
        "messages": [HumanMessage(content="Please keep replies brief and bulleted.")],
    }
    config = {"configurable": {"user_id": "bob"}}

    update = update_profile_node(state, config=config)
    saved = update["user_profile"]

    # New keys merge in, the pre-existing language preference survives.
    assert saved.preferences["tone"] == "brief"
    assert saved.preferences["format"] == "bullets"
    assert saved.preferences["language"] == "en"


def test_update_profile_node_does_not_overwrite_existing_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """If the profile already has a name, a re-introduction does not overwrite it.

    This is an intentional conservative behaviour: once the user has
    told us their name, we hold onto it. A future "actually call me X"
    would need an explicit profile-update flow.
    """

    monkeypatch.setenv("PROFILE_DIR", str(tmp_path))
    fake = _llm_returning('{"name": "Different", "preferences": {}}')
    monkeypatch.setattr("csa_agent.nodes.get_llm", lambda *_a, **_kw: fake)

    profile = UserProfile(user_id="carol", name="Carol")
    state: dict[str, Any] = {
        "user_profile": profile,
        "messages": [HumanMessage(content="My name is Different.")],
    }
    update = update_profile_node(state, config={"configurable": {"user_id": "carol"}})

    assert update["user_profile"].name == "Carol"


def test_update_profile_node_skips_extraction_on_llm_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """An LLM failure during extraction must not block the topic counters."""

    monkeypatch.setenv("PROFILE_DIR", str(tmp_path))
    boom = FakeChatModel(raise_on_call=RuntimeError)
    monkeypatch.setattr("csa_agent.nodes.get_llm", lambda *_a, **_kw: boom)

    profile = UserProfile(user_id="dan")
    # Build a synthetic AIMessage with a tool_call so the topic counter
    # has something to record. After update_profile_node runs, the
    # topic must be present and the profile saved despite the
    # extractor blowing up.
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "filter_by_category", "args": {"category": "REFUND"}, "id": "c1"}],
    )
    state: dict[str, Any] = {
        "user_profile": profile,
        "messages": [HumanMessage(content="show me refunds"), ai],
    }
    update = update_profile_node(state, config={"configurable": {"user_id": "dan"}})

    saved = update["user_profile"]
    assert saved.topic_counts.get("REFUND") == 1
    # No name was set because extraction failed; no exception bubbled up.
    assert saved.name is None
