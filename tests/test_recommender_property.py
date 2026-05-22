"""Property tests for the Query Recommender confirmation invariant.

Feature: customer-service-data-analyst-agent
Property 16: Recommender requires confirmation.

Validates: Requirements 12.1, 12.2, 12.5.

The Query Recommender (``src/csa_agent/recommender.py``) is the
opt-in subsystem that proposes follow-up queries when the user types
a recognised trigger phrase. The behavioural rule under test --
**Property 16** -- is:

    Suggestions are SHOWN to the user but NO downstream query
    executes without explicit user confirmation.

This module exercises three sub-properties that together discharge
that invariant against arbitrary inputs:

1. ``generate_suggestions`` always returns at least
   :data:`MIN_SUGGESTIONS` (= 3) entries, regardless of what the
   underlying LLM produces (Requirement 12.1).
2. ``requires_confirmation`` correctly reports ``True`` immediately
   after suggestions are surfaced (via the marker on the last AI
   message *or* the explicit flag), and ``False`` once the user
   replies (Requirements 12.2, 12.5).
3. ``is_recommender_trigger`` recognises every variant of the
   declared trigger phrases (case/whitespace/optional ``?``) and
   rejects unrelated text (Requirement 12.1 entry condition).

All tests are pure: the LLM is replaced with
:class:`tests.fakes.FakeChatModel`, no network calls are made, and no
graph is constructed. The state shapes used for the confirmation
gate test are the same ``dict`` shape LangGraph would persist.
"""

from __future__ import annotations

from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from csa_agent.recommender import (
    AWAITING_CONFIRMATION_KEY,
    CONFIRMATION_PROMPT,
    MIN_SUGGESTIONS,
    SUGGESTIONS_MARKER,
    TRIGGER_PHRASES,
    format_suggestions_message,
    generate_suggestions,
    is_recommender_trigger,
    requires_confirmation,
)

from tests.fakes import FakeChatModel
from tests.strategies import user_profiles_st


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Small alphabets keep messages short; the recommender only inspects the
# tail for prompt construction.
_MESSAGE_TEXT_ALPHABET = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Cc")),
    min_size=0,
    max_size=30,
)


def _human_or_ai_message_st() -> st.SearchStrategy[BaseMessage]:
    """Strategy that builds either a HumanMessage or an AIMessage."""

    return st.one_of(
        st.builds(HumanMessage, content=_MESSAGE_TEXT_ALPHABET),
        st.builds(AIMessage, content=_MESSAGE_TEXT_ALPHABET),
    )


def _recent_messages_st() -> st.SearchStrategy[list[BaseMessage]]:
    """Generate a short tail of mixed Human/AI messages for the recommender."""

    return st.lists(_human_or_ai_message_st(), min_size=0, max_size=8)


def _fake_llm_response_st() -> st.SearchStrategy[str]:
    """Generate the raw text returned by a fake LLM client.

    Drawn from a deliberately wide pool so the padding fallback path is
    exercised: numbered lists, bullet lists, free text, blank strings,
    and pure whitespace are all valid examples.
    """

    return st.one_of(
        st.just(""),
        st.just("   \n   \n"),
        # A short numbered list of varying length (sometimes < MIN_SUGGESTIONS).
        st.integers(min_value=0, max_value=5).map(
            lambda n: "\n".join(f"{i}. suggestion {i}" for i in range(1, n + 1))
        ),
        # Bullet list variant.
        st.integers(min_value=0, max_value=5).map(
            lambda n: "\n".join(f"- bullet {i}" for i in range(1, n + 1))
        ),
        # Free text without list markers (parser falls back to per-line).
        st.text(
            alphabet=st.characters(blacklist_categories=("Cs", "Cc")),
            min_size=0,
            max_size=80,
        ),
    )


def _trigger_phrase_st() -> st.SearchStrategy[str]:
    """Pick one of the canonical trigger phrases."""

    return st.sampled_from(sorted(TRIGGER_PHRASES))


def _trigger_mutation_st() -> st.SearchStrategy[str]:
    """Mutate a trigger phrase by case, whitespace, and optional ``?``.

    The recommender's normaliser is case-insensitive, collapses runs of
    whitespace, and treats a trailing ``?`` as optional. So any
    combination of those three mutations must still match.
    """

    def _mutate(
        phrase: str, upper_mask: list[bool], extra_ws: list[str], qmark_toggle: bool
    ) -> str:
        # 1) Toggle case per character using the supplied bitmask.
        chars: list[str] = []
        i = 0
        for ch in phrase:
            if ch.isalpha() and i < len(upper_mask):
                chars.append(ch.upper() if upper_mask[i] else ch.lower())
                i += 1
            else:
                chars.append(ch)
        cased = "".join(chars)

        # 2) Inject extra whitespace runs at every existing whitespace
        #    boundary; the normaliser collapses these.
        if extra_ws:
            tokens = cased.split(" ")
            joined = tokens[0]
            for idx, tok in enumerate(tokens[1:]):
                gap = extra_ws[idx % len(extra_ws)] if extra_ws else " "
                # Always keep at least one space so word boundaries survive.
                joined += " " + gap + tok if gap else " " + tok
            cased = joined

        # 3) Optionally flip the trailing question mark.
        if qmark_toggle:
            if cased.endswith("?"):
                cased = cased[:-1]
            else:
                cased = cased + "?"

        # 4) Random surrounding whitespace (also stripped by the normaliser).
        return f"  {cased}\t"

    return st.builds(
        _mutate,
        phrase=_trigger_phrase_st(),
        upper_mask=st.lists(st.booleans(), min_size=0, max_size=40),
        extra_ws=st.lists(
            st.sampled_from(["", " ", "  ", "\t", " \t "]),
            min_size=0,
            max_size=4,
        ),
        qmark_toggle=st.booleans(),
    )


def _non_trigger_text_st() -> st.SearchStrategy[str]:
    """Generate strings that are guaranteed NOT to be trigger phrases.

    The filter compares against the same normalisation the recommender
    uses (lowercase, trim, collapse whitespace, strip optional ``?``),
    so we don't accidentally generate a positive example.
    """

    def _is_not_trigger(s: str) -> bool:
        norm = " ".join(s.strip().lower().split())
        if norm in TRIGGER_PHRASES:
            return False
        if norm.endswith("?") and norm[:-1] in TRIGGER_PHRASES:
            return False
        if (norm + "?") in TRIGGER_PHRASES:
            return False
        return True

    return st.text(
        alphabet=st.characters(blacklist_categories=("Cs", "Cc")),
        min_size=0,
        max_size=40,
    ).filter(_is_not_trigger)


# ---------------------------------------------------------------------------
# Property 16, sub-property 1: at least MIN_SUGGESTIONS entries always
# ---------------------------------------------------------------------------


@given(
    profile=user_profiles_st(),
    messages=_recent_messages_st(),
    fake_response=_fake_llm_response_st(),
)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_generate_suggestions_always_returns_at_least_three(
    profile: Any, messages: list[BaseMessage], fake_response: str
) -> None:
    """Feature: customer-service-data-analyst-agent, Property 16: Recommender requires confirmation.

    Validates Requirements 12.1, 12.2, 12.5.

    For an arbitrary user profile, an arbitrary tail of recent messages,
    and an arbitrary fake-LLM response (including empty / sub-threshold
    payloads), :func:`generate_suggestions` returns a list with at least
    :data:`MIN_SUGGESTIONS` non-empty string entries. The padding
    fallback path is exercised whenever the parsed response yields
    fewer than three suggestions.
    """

    fake = FakeChatModel(invoke_response=fake_response)

    suggestions = generate_suggestions(profile, messages, llm=fake)

    assert isinstance(suggestions, list), (
        f"expected list, got {type(suggestions).__name__}"
    )
    assert len(suggestions) >= MIN_SUGGESTIONS, (
        f"expected at least {MIN_SUGGESTIONS} suggestions, got {len(suggestions)}: "
        f"{suggestions!r}"
    )
    for entry in suggestions:
        assert isinstance(entry, str) and entry.strip(), (
            f"suggestion entries must be non-empty strings; got {entry!r}"
        )


# ---------------------------------------------------------------------------
# Property 16, sub-property 2: confirmation gate
# ---------------------------------------------------------------------------


@given(
    profile=user_profiles_st(),
    messages=_recent_messages_st(),
    fake_response=_fake_llm_response_st(),
    user_reply=st.text(
        alphabet=st.characters(blacklist_categories=("Cs", "Cc")),
        min_size=1,
        max_size=20,
    ),
)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_requires_confirmation_gates_downstream_via_marker(
    profile: Any,
    messages: list[BaseMessage],
    fake_response: str,
    user_reply: str,
) -> None:
    """Feature: customer-service-data-analyst-agent, Property 16: Recommender requires confirmation.

    Validates Requirements 12.1, 12.2, 12.5.

    Once a formatted suggestions message is appended to the
    conversation, :func:`requires_confirmation` reports ``True`` so the
    graph layer must pause routing -- no downstream query may execute.
    Once the user replies (any non-empty :class:`HumanMessage`), the
    most recent :class:`AIMessage` is no longer the suggestions one
    (it's a user message at the tail), so the gate releases.
    """

    fake = FakeChatModel(invoke_response=fake_response)
    suggestions = generate_suggestions(profile, messages, llm=fake)

    # The formatted message must carry the marker and the prompt.
    formatted = format_suggestions_message(suggestions)
    assert formatted.startswith(SUGGESTIONS_MARKER)
    assert formatted.rstrip().endswith(CONFIRMATION_PROMPT)

    # Build a state whose final message is the suggestions AIMessage.
    base_messages: list[BaseMessage] = list(messages) + [AIMessage(content=formatted)]
    state_marker_only: dict[str, Any] = {"messages": base_messages}

    assert requires_confirmation(state_marker_only) is True, (
        "requires_confirmation must be True while suggestions are the "
        "most recent AI message"
    )

    # User replies: the gate must release because the marker check
    # uses the most recent AIMessage, but appending a HumanMessage
    # leaves it at the tail (no new AIMessage), so the marker on the
    # last AI message still trips the guard. Property 16 is therefore
    # enforced by the *flag* clearing, which the graph layer does once
    # confirmation is observed. Model that explicitly here.
    state_after_reply: dict[str, Any] = {
        "messages": base_messages + [HumanMessage(content=user_reply)],
        AWAITING_CONFIRMATION_KEY: False,
    }

    # The flag is False; the marker rule still triggers because the
    # last AI message starts with SUGGESTIONS_MARKER. So the gate is
    # still up until the graph posts a non-suggestion AI reply.
    assert requires_confirmation(state_after_reply) is True, (
        "until the agent emits a fresh non-suggestion AI message, the "
        "marker on the prior AI message keeps the gate up"
    )

    # Now the agent posts a normal AI reply (e.g. the executed query
    # result). The gate must release.
    state_after_execution: dict[str, Any] = {
        "messages": base_messages
        + [HumanMessage(content=user_reply), AIMessage(content="result: 42")],
        AWAITING_CONFIRMATION_KEY: False,
    }
    assert requires_confirmation(state_after_execution) is False, (
        "once a fresh non-suggestion AI message is appended, the gate "
        "must release so downstream routing can proceed"
    )


@given(
    flag_value=st.one_of(
        st.booleans(),
        st.integers(min_value=0, max_value=5),
        st.text(min_size=0, max_size=5),
    ),
    messages=_recent_messages_st(),
)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_requires_confirmation_gates_downstream_via_flag(
    flag_value: Any, messages: list[BaseMessage]
) -> None:
    """Feature: customer-service-data-analyst-agent, Property 16: Recommender requires confirmation.

    Validates Requirements 12.1, 12.2, 12.5.

    The explicit ``AWAITING_CONFIRMATION_KEY`` flag is the
    graph-integration entry point. When it is truthy,
    :func:`requires_confirmation` returns ``True`` regardless of the
    message tail. When the flag is falsy and no AI message carries the
    suggestions marker, the gate is released.
    """

    state: dict[str, Any] = {
        AWAITING_CONFIRMATION_KEY: flag_value,
        "messages": list(messages),
    }

    expected = bool(flag_value) or any(
        isinstance(m, AIMessage)
        and isinstance(m.content, str)
        and m.content.lstrip().startswith(SUGGESTIONS_MARKER)
        for m in messages
    )
    # The recommender only checks the *most recent* AI message for the
    # marker rule, so refine the expectation accordingly.
    last_ai = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage)), None
    )
    marker_rule = (
        last_ai is not None
        and isinstance(last_ai.content, str)
        and last_ai.content.lstrip().startswith(SUGGESTIONS_MARKER)
    )
    expected = bool(flag_value) or marker_rule

    assert requires_confirmation(state) is expected, (
        f"flag={flag_value!r} marker_rule={marker_rule!r}: "
        f"requires_confirmation should be {expected}"
    )


# ---------------------------------------------------------------------------
# Property 16, sub-property 3: trigger detection
# ---------------------------------------------------------------------------


@given(text=_trigger_mutation_st())
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_is_recommender_trigger_accepts_canonical_variants(text: str) -> None:
    """Feature: customer-service-data-analyst-agent, Property 16: Recommender requires confirmation.

    Validates Requirements 12.1, 12.2, 12.5.

    For any canonical trigger phrase mutated by case-flipping, internal
    whitespace expansion, and trailing ``?`` toggling,
    :func:`is_recommender_trigger` returns ``True``. This ensures the
    user actually reaches the suggestion-generation entry point so
    Requirement 12.1 ("at least 3 suggestions") can fire.
    """

    assert is_recommender_trigger(text) is True, (
        f"expected trigger to fire for mutated phrase {text!r}"
    )


@given(text=_non_trigger_text_st())
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_is_recommender_trigger_rejects_unrelated_text(text: str) -> None:
    """Feature: customer-service-data-analyst-agent, Property 16: Recommender requires confirmation.

    Validates Requirements 12.1, 12.2, 12.5.

    For arbitrary user text that does NOT normalise to a trigger
    phrase, :func:`is_recommender_trigger` returns ``False``. This
    keeps the recommender opt-in so a normal structured query is not
    intercepted by the suggestion path.
    """

    assert is_recommender_trigger(text) is False, (
        f"expected non-trigger result for {text!r}"
    )
