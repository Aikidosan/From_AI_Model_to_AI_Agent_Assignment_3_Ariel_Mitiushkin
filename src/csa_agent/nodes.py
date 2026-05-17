"""Graph nodes for the Customer Service Data Analyst Agent.

This module implements the non-router, non-react graph nodes plus the
shared :class:`AgentState` definition that ``graph.py`` will build upon.

Nodes implemented here:

- :func:`decline_node` -- emits the canonical refusal AIMessage for
  out-of-scope queries. Makes zero LLM and zero tool calls so the decline
  path stays pure (Requirement 2.2; Property 8).
- :func:`summarize_node` -- a small ReAct subgraph bound to a curated
  subset of structured tools (``count_rows``, ``show_examples``,
  ``get_intent_distribution``) plus a system prompt instructing the
  model to ground its summary in dataset facts (Requirement 2.4).
- :func:`load_user_profile_node` -- reads ``user_id`` from the
  LangGraph runnable config and injects the loaded :class:`UserProfile`
  into state (Requirements 7.1, 7.6).
- :func:`update_profile_node` -- runs after every turn: scans the
  turn's tool calls for category/intent arguments, records each as a
  topic, and persists the profile. Failures are logged but never block
  the response (Requirement 7.4; design note: "Profile saves are
  best-effort and never block the main response").

State shape:

The :class:`AgentState` ``TypedDict`` mirrors the design's "LangGraph
State" subsection. ``total=False`` is used so node return values can
populate only the fields they actually update -- LangGraph merges
partial updates per its reducer rules.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from .dataset import get_dataset
from .llm import get_llm
from .profile import UserProfile, load_profile, record_topic, save_profile
from .router import RouteLabel
from .tools.core import build_tools

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical refusal text emitted by :func:`decline_node`. Wording matches
#: the design document verbatim so the decline message is stable across
#: refactors and easy to assert against in tests.
CANONICAL_REFUSAL: str = (
    "I can only answer questions about the Bitext Customer Service training "
    "dataset (categories, intents, utterances, counts, and summaries). "
    "I won't answer that out-of-scope question."
)


#: Names of the structured tools exposed to the summarization sub-agent.
#: The summarize path is intentionally narrower than the full ReAct path:
#: it must be able to fetch grounding facts (counts, examples, intent
#: distribution) but should not, for example, recursively invoke
#: ``summarize_category`` (which would make this node call itself).
_SUMMARIZE_TOOL_NAMES: frozenset[str] = frozenset(
    {"count_rows", "show_examples", "get_intent_distribution"}
)


#: System prompt for the summarization sub-agent. Requires the model to
#: ground its narrative in tool observations rather than world knowledge,
#: implementing Requirement 2.4's "ground the summary in dataset facts"
#: clause at the prompt level.
_SUMMARIZE_SYSTEM_PROMPT: str = (
    "You are a data analyst summarizing the Bitext Customer Service training "
    "dataset. Produce a narrative summary grounded in dataset facts. Use the "
    "provided tools (count_rows, show_examples, get_intent_distribution) to "
    "fetch exact counts and representative example utterances before "
    "describing trends. Do not invent details, statistics, or behaviours "
    "that are not supported by tool results. Keep the final answer concise "
    "(3-5 sentences)."
)


# ---------------------------------------------------------------------------
# Shared graph state
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """LangGraph state shared across all nodes.

    Mirrors the design's "LangGraph State" subsection. ``total=False`` so
    node return values may carry only the fields they actually update;
    LangGraph merges partial updates via the configured reducers.

    Fields:
        messages: Conversation history. Reduced with
            :func:`langgraph.graph.message.add_messages` so node returns
            are appended rather than replacing the list.
        route: The label produced by the query router for the current
            turn, or ``None`` before classification has run.
        user_profile: Per-user profile loaded by
            :func:`load_user_profile_node` and updated by
            :func:`update_profile_node`.
        iterations: Counter incremented per ReAct tool call; capped at
            :data:`csa_agent.config.Settings.max_iterations` (15) by the
            graph layer.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    route: RouteLabel | None
    user_profile: UserProfile
    iterations: int


# ---------------------------------------------------------------------------
# Decline node
# ---------------------------------------------------------------------------


def decline_node(state: AgentState) -> dict[str, Any]:
    """Append the canonical refusal AIMessage; make zero LLM/tool calls.

    Implements Requirement 2.2: the out-of-scope branch must not consult
    the LLM's general knowledge. Returning a static AIMessage keeps the
    decline path observable as "no model call, no tool call" which is
    the invariant Property 8 verifies.

    Args:
        state: The current :class:`AgentState`. Unused -- kept in the
            signature so this function is a drop-in LangGraph node.

    Returns:
        A partial state update appending one :class:`AIMessage` with the
        canonical refusal text.
    """

    # ``state`` is intentionally unused; reference it once to keep linters
    # quiet without affecting behaviour.
    del state
    return {"messages": [AIMessage(content=CANONICAL_REFUSAL)]}


# ---------------------------------------------------------------------------
# Summarize node (small ReAct subgraph)
# ---------------------------------------------------------------------------


# Module-level cache for the summarization sub-agent. The sub-agent is
# constructed lazily on first call so importing this module does not
# eagerly load the dataset or instantiate an LLM client. Tests that
# need a different sub-agent can patch ``_summarize_subagent`` directly
# or call :func:`reset_summarize_subagent_cache`.
_summarize_subagent: Any = None


def _build_summarize_subagent() -> Any:
    """Construct the ReAct sub-agent used by :func:`summarize_node`.

    Tools are restricted to the names in :data:`_SUMMARIZE_TOOL_NAMES` so
    the sub-agent can ground its summary in dataset facts without
    delegating back to the LLM-backed ``summarize_category`` tool (which
    would be circular).
    """

    # Local import: ``langgraph.prebuilt`` pulls in optional deps and we
    # only need it when the unstructured branch actually runs.
    from langgraph.prebuilt import create_react_agent

    df = get_dataset()
    all_tools = build_tools(df)
    summarize_tools = [t for t in all_tools if t.name in _SUMMARIZE_TOOL_NAMES]

    # ``prompt`` is the modern parameter name in ``create_react_agent``; in
    # older releases the same role was filled by ``state_modifier``. Either
    # accepts a string that becomes the system instruction for the
    # sub-agent's LLM call.
    return create_react_agent(
        model=get_llm(),
        tools=summarize_tools,
        prompt=_SUMMARIZE_SYSTEM_PROMPT,
    )


def _get_summarize_subagent() -> Any:
    """Return the cached summarization sub-agent, building it on first use."""

    global _summarize_subagent
    if _summarize_subagent is None:
        _summarize_subagent = _build_summarize_subagent()
    return _summarize_subagent


def reset_summarize_subagent_cache() -> None:
    """Drop the cached sub-agent so the next call rebuilds it.

    Intended for tests that swap LLM doubles between cases.
    """

    global _summarize_subagent
    _summarize_subagent = None


def summarize_node(state: AgentState) -> dict[str, Any]:
    """Invoke the summarization ReAct sub-agent over the conversation.

    The sub-agent receives the full conversation history so it can resolve
    references like *"summarize that category"*. Only the *new* messages
    produced by the sub-agent are returned to the parent graph, so the
    parent's ``add_messages`` reducer appends them without re-emitting
    the inputs.

    Args:
        state: The current :class:`AgentState`. Only ``messages`` is
            consumed.

    Returns:
        A partial state update of the form ``{"messages": <new_messages>}``
        where ``<new_messages>`` is the delta produced by the sub-agent
        (typically tool calls, tool messages, and a final AIMessage).
    """

    subagent = _get_summarize_subagent()
    messages: list[BaseMessage] = list(state.get("messages", []))
    result = subagent.invoke({"messages": messages})

    # ``create_react_agent`` returns the full message history. Extract just
    # the messages it added so the parent graph's ``add_messages`` reducer
    # appends them instead of re-emitting (and potentially duplicating)
    # the inputs.
    full_messages: list[BaseMessage] = list(result.get("messages", []))
    delta = full_messages[len(messages):]
    return {"messages": delta}


# ---------------------------------------------------------------------------
# Profile lifecycle nodes
# ---------------------------------------------------------------------------


_DEFAULT_USER_ID: str = "default"


def _user_id_from_config(config: dict[str, Any] | None) -> str:
    """Extract ``user_id`` from a LangGraph runnable config.

    Falls back to ``"default"`` when the key is absent or empty so the
    agent works without a ``--user`` flag (Requirement 7.6).
    """

    if not config:
        return _DEFAULT_USER_ID
    configurable = config.get("configurable") or {}
    user_id = configurable.get("user_id")
    if isinstance(user_id, str) and user_id.strip():
        return user_id.strip()
    return _DEFAULT_USER_ID


def load_user_profile_node(
    state: AgentState,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the per-user profile and inject it into state.

    Implements the ``load_user_profile`` node from the design's node graph
    (Requirements 7.1, 7.6). Reads ``user_id`` from
    ``config["configurable"]["user_id"]``, defaulting to ``"default"``
    when absent, and returns the loaded :class:`UserProfile` as a
    partial state update.

    Args:
        state: The current :class:`AgentState`. Unused.
        config: LangGraph runnable config. Expected shape:
            ``{"configurable": {"user_id": "<id>", "thread_id": "<id>"}}``.

    Returns:
        ``{"user_profile": <UserProfile>}``.
    """

    del state  # not consumed
    user_id = _user_id_from_config(config)
    profile = load_profile(user_id)
    return {"user_profile": profile}


def _topics_in_current_turn(messages: list[BaseMessage]) -> set[str]:
    """Return category/intent topic strings referenced in the current turn.

    "Current turn" is defined as the message slice starting at the last
    :class:`HumanMessage`. Limiting extraction to that slice avoids
    re-counting tool calls from earlier turns each time
    :func:`update_profile_node` runs, which would otherwise inflate
    ``topic_counts`` beyond what the user actually queried this turn.

    The function inspects ``tool_calls`` on AIMessages (LangChain's
    standard attribute name) and pulls the ``category`` and ``intent``
    arguments from each call. Both dict-shaped tool calls (the modern
    LangChain representation) and object-shaped ones are handled.
    """

    if not messages:
        return set()

    # Find the last HumanMessage to mark the start of the current turn.
    last_human_idx: int = -1
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            last_human_idx = i
    turn_messages = (
        messages[last_human_idx:] if last_human_idx >= 0 else list(messages)
    )

    topics: set[str] = set()
    for msg in turn_messages:
        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            if isinstance(tc, dict):
                args = tc.get("args") or {}
            else:
                args = getattr(tc, "args", {}) or {}
            if not isinstance(args, dict):
                continue
            for key in ("category", "intent"):
                value = args.get(key)
                if isinstance(value, str) and value.strip():
                    topics.add(value.strip())
    return topics


def update_profile_node(
    state: AgentState,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record turn topics and persist the profile.

    Behaviour:

    * Pulls the in-state :class:`UserProfile` (loaded by
      :func:`load_user_profile_node`); falls back to a fresh load by
      ``user_id`` if state has not been hydrated.
    * Extracts category/intent argument values from this turn's tool
      calls via :func:`_topics_in_current_turn` and calls
      :func:`record_topic` for each unique value.
    * Persists the updated profile via :func:`save_profile`.
    * Wraps the whole sequence in a broad ``except`` that logs a warning
      and returns an empty update, so a profile-store hiccup never
      blocks the user's answer (design: "Profile saves are best-effort
      and never block the main response").

    Args:
        state: The current :class:`AgentState`. ``messages`` and
            ``user_profile`` are consumed.
        config: LangGraph runnable config; used only to resolve the
            ``user_id`` fallback when ``user_profile`` is missing from
            state.

    Returns:
        ``{"user_profile": <saved_profile>}`` on success, or ``{}`` when
        the update fails. Returning ``{}`` leaves the existing state
        ``user_profile`` untouched.
    """

    try:
        profile = state.get("user_profile") if isinstance(state, dict) else None
        if profile is None:
            # State hasn't been hydrated (e.g. tests invoking this node in
            # isolation). Load by user_id so the topic counters survive.
            profile = load_profile(_user_id_from_config(config))

        topics = _topics_in_current_turn(list(state.get("messages", [])))
        for topic in topics:
            record_topic(profile, topic)

        saved = save_profile(profile)
        return {"user_profile": saved}
    except Exception:  # noqa: BLE001 - profile failures must not block the turn
        # Log with the offending user_id when we can recover it; otherwise
        # log a placeholder so the warning is still actionable.
        offending_user_id: str
        try:
            offending_user_id = _user_id_from_config(config)
        except Exception:  # pragma: no cover - defensive
            offending_user_id = "<unknown>"
        logger.warning(
            "update_profile_node: failed to update profile for user_id=%s",
            offending_user_id,
            exc_info=True,
        )
        return {}


__all__ = [
    "AgentState",
    "CANONICAL_REFUSAL",
    "decline_node",
    "load_user_profile_node",
    "reset_summarize_subagent_cache",
    "summarize_node",
    "update_profile_node",
]
