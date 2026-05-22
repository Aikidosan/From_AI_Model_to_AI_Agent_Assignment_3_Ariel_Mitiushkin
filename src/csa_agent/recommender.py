"""Query Recommender (bonus) for the Customer Service Data Analyst Agent.

The Query Recommender is the optional subsystem described by Requirement 12
of the spec. When the user types a recognised trigger phrase such as
*"What should I query next?"*, the agent uses the loaded
:class:`~csa_agent.profile.UserProfile` and the most recent conversation
messages to propose at least three follow-up queries to run against the
Bitext customer service dataset.

This module deliberately ships only the **building blocks** (trigger
detection, suggestion generation, formatted output, confirmation
tracker). Wiring these into the LangGraph node graph is a separate
follow-up task; isolating the pure pieces here keeps them easy to unit
and property test in isolation.

Confirmation invariant (Property 16)
------------------------------------
The single most important behavioural rule of this subsystem -- and the
invariant validated by **Property 16** ("Recommender requires
confirmation") -- is:

    *Suggestions are SHOWN to the user but NO downstream query
    executes without explicit user confirmation.*

Concretely, this means:

* :func:`generate_suggestions` returns a ``list[str]`` of suggestion
  texts. It does **not** invoke any dataset tool, the query router, or
  the ReAct sub-agent.
* :func:`format_suggestions_message` produces a human-readable
  numbered list ending in an explicit confirmation prompt.
* :func:`requires_confirmation` exposes a boolean guard that the graph
  layer can use to short-circuit routing while the user has not yet
  selected, refined, or rejected a suggestion.

The graph integration step (a future task) is responsible for setting
``state["awaiting_recommendation_confirmation"] = True`` after showing
suggestions, and clearing it once the user replies with a number, a
refinement, or a rejection.

Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Final, Iterable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .llm import get_llm
from .profile import UserProfile

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trigger detection
# ---------------------------------------------------------------------------

#: Phrases that activate the Query Recommender. Matching is case- and
#: punctuation-insensitive (see :func:`is_recommender_trigger`). Keep this
#: set narrow on purpose: the recommender is opt-in, and broad triggers
#: would surprise users who simply asked a structured question.
TRIGGER_PHRASES: Final[frozenset[str]] = frozenset(
    {
        "what should i query next",
        "what should i query next?",
        "what should i ask next",
        "what should i ask next?",
        "suggest a query",
        "suggest queries",
        "suggest some queries",
        "recommend a query",
        "recommend queries",
    }
)


def _normalise_trigger(query: str) -> str:
    """Lowercase, trim, and collapse whitespace for trigger comparison."""

    # Collapse runs of whitespace so "what  should  i query next" matches.
    return " ".join(query.strip().lower().split())


def is_recommender_trigger(query: str) -> bool:
    """Return ``True`` iff ``query`` is a recommender trigger phrase.

    Comparison is case-insensitive and tolerant of leading/trailing
    whitespace and internal whitespace runs. A trailing question mark is
    accepted but not required.

    Args:
        query: The raw user query text.

    Returns:
        ``True`` when ``query`` (after normalisation) matches one of
        :data:`TRIGGER_PHRASES`; ``False`` otherwise.
    """

    if not isinstance(query, str):
        return False
    normalised = _normalise_trigger(query)
    if not normalised:
        return False
    # Match either the exact normalised form or the form without a
    # trailing question mark, so we don't have to enumerate every
    # punctuation variant in TRIGGER_PHRASES.
    if normalised in TRIGGER_PHRASES:
        return True
    if normalised.endswith("?") and normalised[:-1] in TRIGGER_PHRASES:
        return True
    if (normalised + "?") in TRIGGER_PHRASES:
        return True
    return False


# ---------------------------------------------------------------------------
# Suggestion generation
# ---------------------------------------------------------------------------

#: Minimum number of suggestions surfaced to the user (Requirement 12.1).
MIN_SUGGESTIONS: Final[int] = 3

#: Default suggestions used to pad short LLM responses up to
#: :data:`MIN_SUGGESTIONS`. Chosen to be safe, dataset-grounded, and
#: representative of the three core query shapes (list / distribution /
#: summary) the agent supports.
DEFAULT_SUGGESTIONS: Final[tuple[str, ...]] = (
    "List all categories",
    "Show the distribution of intents in REFUND",
    "Summarize the FEEDBACK category",
    "Show 5 examples from the ORDER category",
    "Count rows in the CONTACT category",
)


_SUGGEST_SYSTEM_PROMPT: Final[str] = (
    "You are the Query Recommender for a Customer Service Data Analyst "
    "Agent that answers questions about the Bitext Customer Service "
    "Tagged Training Dataset. The dataset contains customer utterances "
    "labelled with a category (e.g. REFUND, ORDER, FEEDBACK, CONTACT) "
    "and an intent (e.g. track_refund, cancel_order).\n\n"
    "The agent supports these query shapes:\n"
    "  - list categories\n"
    "  - count rows (optionally filtered by category and/or intent)\n"
    "  - filter by category or intent\n"
    "  - show N example utterances from a category or intent\n"
    "  - show the distribution of intents within a category\n"
    "  - summarize a category in natural language\n\n"
    "Given the user's profile and recent conversation, propose at least "
    "three concrete follow-up queries the user could ask next. Prefer "
    "queries that build on what was just discussed, broaden coverage to "
    "categories/intents the user has not yet explored, or compare across "
    "categories. Each suggestion must be a single, self-contained, "
    "imperative sentence the user could literally send as their next "
    "message. Do not invent categories or intents that are not part of a "
    "customer-service taxonomy.\n\n"
    "Respond as a numbered list (1., 2., 3., ...). No preamble, no "
    "trailing commentary -- only the numbered list."
)


def _profile_summary(profile: UserProfile) -> str:
    """Render a compact textual summary of ``profile`` for the prompt.

    Keeps the prompt budget small while exposing the three personalisation
    signals the recommender cares about: name, frequent topics, and free
    form preferences.
    """

    parts: list[str] = [f"user_id: {profile.user_id}"]
    if profile.name:
        parts.append(f"name: {profile.name}")
    if profile.frequent_topics:
        parts.append(
            "frequent_topics: " + ", ".join(profile.frequent_topics)
        )
    if profile.preferences:
        prefs = "; ".join(f"{k}={v}" for k, v in profile.preferences.items())
        parts.append(f"preferences: {prefs}")
    return "\n".join(parts)


def _excerpt_message(msg: BaseMessage, max_chars: int = 240) -> str:
    """Return ``role: text`` excerpt for ``msg``, truncated to ``max_chars``."""

    if isinstance(msg, HumanMessage):
        role = "user"
    elif isinstance(msg, AIMessage):
        role = "assistant"
    elif isinstance(msg, SystemMessage):
        role = "system"
    else:
        role = getattr(msg, "type", msg.__class__.__name__) or "message"

    content = msg.content
    if isinstance(content, list):
        # LangChain occasionally packs multimodal content as a list of
        # parts. Flatten to plain text for prompt purposes.
        flattened: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content") or ""
                if text:
                    flattened.append(str(text))
            elif part is not None:
                flattened.append(str(part))
        content = " ".join(flattened)
    text = (str(content) if content is not None else "").strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return f"{role}: {text}" if text else f"{role}: <empty>"


def _format_recent_messages(
    messages: Iterable[BaseMessage], limit: int = 8
) -> str:
    """Return a newline-joined excerpt of the most recent ``limit`` messages."""

    items = [m for m in messages if m is not None]
    if not items:
        return "(no prior conversation)"
    tail = items[-limit:]
    return "\n".join(_excerpt_message(m) for m in tail)


# Pattern for stripping a leading "1.", "2)", "- ", "* ", "• " bullet/number.
_LIST_PREFIX_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:\d+\s*[\.\)\-:]|[-*•])\s+"
)


def _parse_suggestions(raw: str) -> list[str]:
    """Parse an LLM response into a list of suggestion strings.

    Recognises numbered lists (``1. ...``, ``2) ...``) as well as
    bullet lists (``- ...``, ``* ...``, ``• ...``). Lines that do not
    begin with a list marker are ignored, except as a fallback when no
    list markers are found (in which case each non-empty line is
    treated as a suggestion).
    """

    if not isinstance(raw, str) or not raw.strip():
        return []

    lines = [ln.strip() for ln in raw.splitlines()]
    suggestions: list[str] = []
    for line in lines:
        if not line:
            continue
        match = _LIST_PREFIX_RE.match(line)
        if match:
            suggestion = line[match.end():].strip()
            if suggestion:
                suggestions.append(suggestion)

    if suggestions:
        return suggestions

    # Fallback: treat every non-empty line as a suggestion. This rescues
    # responses that omit list markers despite our prompt asking for them.
    return [ln for ln in lines if ln]


def _pad_with_defaults(suggestions: list[str], minimum: int) -> list[str]:
    """Pad ``suggestions`` up to ``minimum`` entries using defaults.

    Defaults are appended in their declared order, skipping any that are
    already present (case-insensitive) so we never duplicate a suggestion
    the LLM already produced.
    """

    if len(suggestions) >= minimum:
        return suggestions

    seen_lower = {s.strip().lower() for s in suggestions}
    padded = list(suggestions)
    for default in DEFAULT_SUGGESTIONS:
        if len(padded) >= minimum:
            break
        if default.strip().lower() in seen_lower:
            continue
        padded.append(default)
        seen_lower.add(default.strip().lower())

    # If DEFAULT_SUGGESTIONS is somehow exhausted without reaching the
    # threshold (shouldn't happen, but defensive), top up with a generic
    # filler so the post-condition len(result) >= minimum always holds.
    while len(padded) < minimum:
        padded.append("List all categories")
    return padded


def generate_suggestions(
    profile: UserProfile,
    recent_messages: list[BaseMessage],
    llm: "ChatOpenAI | None" = None,
) -> list[str]:
    """Generate at least :data:`MIN_SUGGESTIONS` follow-up query suggestions.

    Builds a focused prompt from the user profile and a tail of the recent
    conversation, asks the configured Nebius LLM for a numbered list, and
    parses the response. If parsing yields fewer than
    :data:`MIN_SUGGESTIONS` items (or the LLM call fails), the result is
    padded with entries from :data:`DEFAULT_SUGGESTIONS` so the caller
    can always honour Requirement 12.1.

    This function is **read-only**: it never invokes a dataset tool, the
    router, or the ReAct sub-agent. Suggestions are surfaced to the user
    via :func:`format_suggestions_message`; downstream execution must
    wait for explicit confirmation (Requirement 12.5; Property 16).

    Args:
        profile: The active :class:`UserProfile`. Used to personalise
            suggestions (name, frequent topics, preferences).
        recent_messages: A list of :class:`BaseMessage` instances from
            the current session, oldest first. Only the tail is
            included in the prompt.
        llm: Optional pre-constructed chat-model client. When ``None``,
            :func:`csa_agent.llm.get_llm` is called -- preserving the
            single-LLM-factory invariant (Requirement 9 / Property 14).

    Returns:
        A list of at least :data:`MIN_SUGGESTIONS` non-empty suggestion
        strings.
    """

    client: "ChatOpenAI" = llm if llm is not None else get_llm()

    user_prompt = (
        "User profile:\n"
        f"{_profile_summary(profile)}\n\n"
        "Recent conversation (most recent last):\n"
        f"{_format_recent_messages(recent_messages)}\n\n"
        f"Propose at least {MIN_SUGGESTIONS} follow-up queries as a "
        "numbered list."
    )

    suggestions: list[str] = []
    try:
        response = client.invoke(
            [
                ("system", _SUGGEST_SYSTEM_PROMPT),
                ("human", user_prompt),
            ]
        )
        raw_text = getattr(response, "content", response)
        if isinstance(raw_text, list):
            # Flatten multimodal-style content to plain text.
            raw_text = " ".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in raw_text
            )
        suggestions = _parse_suggestions(str(raw_text))
    except Exception:  # noqa: BLE001 - any failure must not bubble to the user
        logger.warning(
            "generate_suggestions: LLM call failed; falling back to defaults",
            exc_info=True,
        )
        suggestions = []

    return _pad_with_defaults(suggestions, MIN_SUGGESTIONS)


# ---------------------------------------------------------------------------
# Presentation and confirmation
# ---------------------------------------------------------------------------

#: Marker prepended to formatted suggestion messages. The graph layer can
#: cheaply detect "this was a suggestions message" by checking
#: :func:`AIMessage.content.startswith(SUGGESTIONS_MARKER)`, which is the
#: minimal coupling needed for :func:`requires_confirmation` to work
#: even when state lacks an explicit flag.
SUGGESTIONS_MARKER: Final[str] = "Here are a few queries you could try next:"

#: Suffix asking the user to confirm or refine. Kept as a constant so
#: tests can assert on its presence (Requirement 12.2 / 12.4).
CONFIRMATION_PROMPT: Final[str] = (
    "Reply with a number to confirm, or refine the query in your own words."
)


def format_suggestions_message(suggestions: list[str]) -> str:
    """Render ``suggestions`` as an AI-message-friendly numbered list.

    The output begins with :data:`SUGGESTIONS_MARKER`, followed by a
    numbered list, and ends with :data:`CONFIRMATION_PROMPT`. The
    confirmation prompt is what makes Requirement 12.2 visible to the
    user: the agent has shown options and is now waiting.

    Args:
        suggestions: The list of suggestion strings to display. Empty
            entries are skipped. The caller is responsible for ensuring
            at least :data:`MIN_SUGGESTIONS` non-empty suggestions are
            present (use :func:`generate_suggestions`, which already
            guarantees this).

    Returns:
        A single string ready to be wrapped in an :class:`AIMessage` and
        appended to the conversation.
    """

    cleaned = [s.strip() for s in suggestions if isinstance(s, str) and s.strip()]
    if not cleaned:
        cleaned = list(DEFAULT_SUGGESTIONS[:MIN_SUGGESTIONS])

    numbered = "\n".join(f"{i}. {text}" for i, text in enumerate(cleaned, start=1))
    return f"{SUGGESTIONS_MARKER}\n{numbered}\n\n{CONFIRMATION_PROMPT}"


# ---------------------------------------------------------------------------
# Confirmation tracker
# ---------------------------------------------------------------------------

#: State key used to indicate the agent is waiting for the user to confirm
#: or refine a suggestion. The graph integration task is responsible for
#: setting this flag after :func:`format_suggestions_message` is appended
#: to the conversation, and clearing it once the user replies.
AWAITING_CONFIRMATION_KEY: Final[str] = "awaiting_recommendation_confirmation"


def _last_ai_message(messages: Iterable[BaseMessage]) -> AIMessage | None:
    """Return the most recent :class:`AIMessage` in ``messages``, if any."""

    last: AIMessage | None = None
    for msg in messages:
        if isinstance(msg, AIMessage):
            last = msg
    return last


def requires_confirmation(state: Any) -> bool:
    """Return ``True`` iff the graph should pause for user confirmation.

    This is the guard that enforces the Property 16 invariant: the graph
    must not invoke the normal routing path while suggestions are
    awaiting a response. The function is intentionally permissive about
    the shape of ``state`` so it can be called from either a TypedDict
    (LangGraph) or a plain ``dict`` test fixture.

    Detection rules (in priority order):

    1. If ``state`` carries the :data:`AWAITING_CONFIRMATION_KEY` flag
       set to a truthy value, return ``True``.
    2. Else, if the most recent :class:`AIMessage` in
       ``state["messages"]`` begins with :data:`SUGGESTIONS_MARKER`,
       return ``True``. This rule lets older state (from before the
       flag was introduced) still be recognised.
    3. Otherwise, return ``False``.

    Args:
        state: A LangGraph state mapping (TypedDict / dict).

    Returns:
        ``True`` when downstream routing must wait for confirmation.
    """

    if not isinstance(state, dict):
        return False

    flag = state.get(AWAITING_CONFIRMATION_KEY)
    if flag:
        return True

    messages = state.get("messages") or []
    if not isinstance(messages, list):
        return False
    last_ai = _last_ai_message(messages)
    if last_ai is None:
        return False
    content = last_ai.content
    if isinstance(content, list):
        # Flatten multimodal content for the marker check.
        content = " ".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    if not isinstance(content, str):
        return False
    return content.lstrip().startswith(SUGGESTIONS_MARKER)


__all__ = [
    "AWAITING_CONFIRMATION_KEY",
    "CONFIRMATION_PROMPT",
    "DEFAULT_SUGGESTIONS",
    "MIN_SUGGESTIONS",
    "SUGGESTIONS_MARKER",
    "TRIGGER_PHRASES",
    "format_suggestions_message",
    "generate_suggestions",
    "is_recommender_trigger",
    "requires_confirmation",
]
