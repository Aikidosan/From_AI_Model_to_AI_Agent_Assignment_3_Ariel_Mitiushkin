"""LangGraph assembly for the Customer Service Data Analyst Agent.

This module wires the design's node graph into a single compiled
:class:`~langgraph.graph.state.CompiledStateGraph`:

.. code-block:: text

    __start__ -> load_user_profile -> query_router
        query_router -[STRUCTURED]----> react_agent
        query_router -[UNSTRUCTURED]--> summarize_node
        query_router -[OUT_OF_SCOPE]--> decline_node
        react_agent ---\
        summarize_node -+-> update_profile -> __end__
        decline_node ---/

The graph is exposed via :func:`build_graph`, a small factory that:

* Accepts an already-loaded :class:`pandas.DataFrame` (defaults to
  :func:`csa_agent.dataset.get_dataset`) and an already-entered
  LangGraph checkpointer (callers manage the ``with`` because
  :func:`csa_agent.checkpointer.get_checkpointer` is a context manager).
* Builds a single ReAct sub-agent up front with a static base system
  prompt, and a node wrapper that prepends a per-call
  :class:`~langchain_core.messages.SystemMessage` carrying the loaded
  user profile so the model can answer "what do you remember about me?".
* Applies the iteration cap by invoking the sub-agent with
  ``config={"recursion_limit": settings.max_iterations}`` and catches
  :class:`langgraph.errors.GraphRecursionError` to emit the graceful
  fallback message required by Requirement 4.3.
* Returns the compiled graph.

For frontends that want to render reasoning steps as they happen,
:func:`stream_graph` is a thin wrapper around ``graph.stream`` that
yields a uniform sequence of ``("tool", name, args, observation)``
events followed by a single ``("final", content)`` event when the run
finishes -- the contract Property 11 (Requirement 4.4) checks.

Validates:

* Requirements 2.3, 2.4 -- routing matches classification.
* Requirements 4.1, 4.2, 4.3, 4.4 -- multi-step reasoning, iteration
  cap, graceful fallback, and tool events streamed before the final.
* Requirements 6.1, 6.4, 6.5 -- compiled with a checkpointer so every
  super-step persists; persistence failures surface to the caller.
"""

from __future__ import annotations

from typing import Any, Iterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from .config import get_settings
from .dataset import get_dataset
from .llm import get_llm
from .nodes import (
    AgentState,
    confirmation_node,
    decline_node,
    load_user_profile_node,
    recommender_node,
    summarize_node,
    update_profile_node,
)
from .profile import UserProfile
from .recommender import is_recommender_trigger
from .router import RouteLabel, classify_query
from .tools.core import build_tools


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Node names used in the graph definition. Centralised so the conditional
#: router and the streaming helper agree on identifiers.
LOAD_PROFILE_NODE: str = "load_user_profile"
QUERY_ROUTER_NODE: str = "query_router"
REACT_AGENT_NODE: str = "react_agent"
SUMMARIZE_NODE: str = "summarize"
DECLINE_NODE: str = "decline"
UPDATE_PROFILE_NODE: str = "update_profile"
RECOMMENDER_NODE: str = "recommender"
CONFIRMATION_NODE: str = "confirmation"


#: Base system prompt for the ReAct sub-agent. Per-call profile context is
#: prepended as a separate :class:`SystemMessage` inside
#: :func:`_react_agent_node` so a single sub-agent instance can be reused
#: across users without rebuilding the graph.
_REACT_BASE_PROMPT: str = (
    "You are a Customer Service Data Analyst Agent. Answer the user's "
    "question by grounding your reasoning in the dataset tools provided. "
    "Always use the tools to fetch exact counts, examples, or distributions "
    "before stating numerical facts. Be concise and direct in your final "
    "answer."
)


#: Message returned when the ReAct sub-agent exceeds ``max_iterations``.
#: Wording matches Requirement 4.3 ("could not be completed within the
#: iteration limit").
_RECURSION_FALLBACK_MESSAGE: str = (
    "I couldn't complete this query within the reasoning step limit. "
    "Try rephrasing the question or breaking it into smaller parts."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _profile_context_message(profile: UserProfile | None) -> SystemMessage:
    """Build a :class:`SystemMessage` describing the loaded user profile.

    The text follows the design's "state_modifier" template so
    "what do you remember about me?" turns can be answered straight from
    the conversation context without an extra tool call (Requirement 7.3).
    """

    name = profile.name if profile and profile.name else "unknown"
    if profile and profile.frequent_topics:
        frequent = ", ".join(profile.frequent_topics)
    else:
        frequent = "none yet"
    if profile and profile.preferences:
        preferences = ", ".join(
            f"{key}={value}" for key, value in profile.preferences.items()
        )
    else:
        preferences = "none recorded"

    text = (
        "User profile context:\n"
        f"- Name: {name}\n"
        f"- Frequent topics: {frequent}\n"
        f"- Preferences: {preferences}\n"
    )
    return SystemMessage(content=text)


def _latest_human_text(messages: list[BaseMessage]) -> str:
    """Return the text of the most recent :class:`HumanMessage`.

    Returns an empty string when no human message is present so the router
    can still produce a deterministic ``OUT_OF_SCOPE`` label rather than
    raising on degenerate inputs.
    """

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            if isinstance(content, str):
                return content
            # Some providers return list-of-content-parts; flatten to text.
            if isinstance(content, list):
                return "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            return str(content)
    return ""


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------


def _make_query_router_node():
    """Return the ``query_router`` node bound to a fresh classification LLM.

    The LLM is captured in the closure so we make a single
    :func:`get_llm` call per :func:`build_graph` invocation rather than
    one per turn.
    """

    classifier_llm = get_llm()

    def query_router_node(state: AgentState) -> dict[str, Any]:
        """Classify the latest user query and write the label into state."""

        user_query = _latest_human_text(list(state.get("messages", [])))
        label = classify_query(user_query, classifier_llm)
        return {"route": label}

    return query_router_node


def _route_from_state(state: AgentState) -> str:
    """Conditional-edge function: pick the next node from ``state['route']``.

    Falls back to the decline path when ``route`` is missing or carries an
    unrecognised value, matching the router's "safe default" posture
    (Requirement 2.2).
    """

    route = state.get("route") if isinstance(state, dict) else None
    if route == RouteLabel.STRUCTURED:
        return REACT_AGENT_NODE
    if route == RouteLabel.UNSTRUCTURED:
        return SUMMARIZE_NODE
    return DECLINE_NODE


def _pre_router_route(state: AgentState) -> str:
    """Conditional-edge function evaluated immediately after profile load.

    Three branches:

    1. **Awaiting confirmation.** When the previous turn ended in a
       recommender suggestion list, ``state['awaiting_recommendation_confirmation']``
       is ``True`` and the latest user message is interpreted as a reply
       (confirm / refine / reject). Route to :func:`confirmation_node`.
    2. **Recommender trigger.** When the latest user message matches a
       recommender trigger phrase such as *"What should I query next?"*,
       route to :func:`recommender_node` to surface suggestions instead
       of running the normal classification pipeline.
    3. **Normal flow.** Anything else routes through the regular query
       router so structured / unstructured / out-of-scope classification
       runs as before.
    """

    if isinstance(state, dict) and state.get("awaiting_recommendation_confirmation"):
        return CONFIRMATION_NODE

    messages = state.get("messages", []) if isinstance(state, dict) else []
    latest_human = _latest_human_text(list(messages))
    if latest_human and is_recommender_trigger(latest_human):
        return RECOMMENDER_NODE

    return QUERY_ROUTER_NODE


def _after_confirmation_route(state: AgentState) -> str:
    """Where to go after :func:`confirmation_node` runs.

    * ``"confirmed"`` -- the user picked a suggestion, which has been
      injected as a fresh ``HumanMessage``. Route through the regular
      query router so the chosen query is classified and executed.
    * ``"refined"`` / ``"rejected"`` -- the turn ends here. Skip the
      regular pipeline and go straight to ``update_profile`` so the
      checkpoint persists and the next user message starts fresh.
    """

    action = state.get("confirmation_action") if isinstance(state, dict) else None
    if action == "confirmed":
        return QUERY_ROUTER_NODE
    return UPDATE_PROFILE_NODE


def _make_react_agent_node(tools: list[BaseTool]) -> Any:
    """Build the ReAct sub-agent and wrap it in a graph node function.

    A single sub-agent instance is constructed up front with the static
    base prompt; per-call profile context is prepended inside the wrapper
    so the same sub-agent works for every user without rebuilding the
    graph.
    """

    settings = get_settings()
    subagent = create_react_agent(
        model=get_llm(),
        tools=tools,
        prompt=_REACT_BASE_PROMPT,
    )

    def react_agent_node(state: AgentState) -> dict[str, Any]:
        """Run the ReAct sub-agent under the configured iteration cap.

        Returns only the *delta* of new messages produced by the
        sub-agent so the parent graph's ``add_messages`` reducer appends
        them rather than re-emitting the inputs. On
        :class:`GraphRecursionError` (the LangGraph signal that the
        recursion limit was reached) returns the graceful fallback
        AIMessage required by Requirement 4.3.
        """

        messages: list[BaseMessage] = list(state.get("messages", []))
        profile_msg = _profile_context_message(state.get("user_profile"))
        sub_input = {"messages": [profile_msg, *messages]}

        try:
            result = subagent.invoke(
                sub_input,
                config={"recursion_limit": settings.max_iterations},
            )
        except GraphRecursionError:
            return {"messages": [AIMessage(content=_RECURSION_FALLBACK_MESSAGE)]}

        full_messages: list[BaseMessage] = list(result.get("messages", []))
        # The sub-agent received ``[profile_msg, *messages]`` as input, so
        # the prefix length we want to skip is ``1 + len(messages)``.
        prefix = 1 + len(messages)
        delta = full_messages[prefix:]
        return {"messages": delta}

    return react_agent_node


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph(
    *,
    df: Any | None = None,
    checkpointer: Any | None = None,
) -> CompiledStateGraph:
    """Assemble and compile the agent graph.

    Args:
        df: Optional pandas DataFrame to bind tools against. When ``None``
            the cached singleton from :func:`csa_agent.dataset.get_dataset`
            is used.
        checkpointer: An already-entered LangGraph checkpointer. Callers
            obtain one from :func:`csa_agent.checkpointer.get_checkpointer`
            inside their own ``with`` block, since that helper is a
            context manager. When ``None``, the graph is compiled without
            a checkpointer (useful for tests that do not need persistence).

    Returns:
        The compiled :class:`CompiledStateGraph` ready for ``invoke`` or
        ``stream``.
    """

    if df is None:
        df = get_dataset()

    tools = build_tools(df)

    builder: StateGraph = StateGraph(AgentState)

    # Register nodes.
    builder.add_node(LOAD_PROFILE_NODE, load_user_profile_node)
    builder.add_node(QUERY_ROUTER_NODE, _make_query_router_node())
    builder.add_node(REACT_AGENT_NODE, _make_react_agent_node(tools))
    builder.add_node(SUMMARIZE_NODE, summarize_node)
    builder.add_node(DECLINE_NODE, decline_node)
    builder.add_node(UPDATE_PROFILE_NODE, update_profile_node)
    builder.add_node(RECOMMENDER_NODE, recommender_node)
    builder.add_node(CONFIRMATION_NODE, confirmation_node)

    # Wire the deterministic edges.
    builder.add_edge(START, LOAD_PROFILE_NODE)

    # Pre-router fan-out: profile load -> {confirmation, recommender, query_router}.
    # When the previous turn ended in a suggestion list, skip the
    # classifier; when the latest user message is a recommender trigger,
    # route to the recommender; otherwise fall through to the normal
    # router.
    builder.add_conditional_edges(
        LOAD_PROFILE_NODE,
        _pre_router_route,
        {
            CONFIRMATION_NODE: CONFIRMATION_NODE,
            RECOMMENDER_NODE: RECOMMENDER_NODE,
            QUERY_ROUTER_NODE: QUERY_ROUTER_NODE,
        },
    )

    # Routing fan-out from the query_router node.
    builder.add_conditional_edges(
        QUERY_ROUTER_NODE,
        _route_from_state,
        {
            REACT_AGENT_NODE: REACT_AGENT_NODE,
            SUMMARIZE_NODE: SUMMARIZE_NODE,
            DECLINE_NODE: DECLINE_NODE,
        },
    )

    # Confirmation fan-out:
    # * confirmed -> route the injected query through the regular pipeline.
    # * refined / rejected -> end the turn after persisting the profile.
    builder.add_conditional_edges(
        CONFIRMATION_NODE,
        _after_confirmation_route,
        {
            QUERY_ROUTER_NODE: QUERY_ROUTER_NODE,
            UPDATE_PROFILE_NODE: UPDATE_PROFILE_NODE,
        },
    )

    # Convergence into the profile-update tail.
    builder.add_edge(REACT_AGENT_NODE, UPDATE_PROFILE_NODE)
    builder.add_edge(SUMMARIZE_NODE, UPDATE_PROFILE_NODE)
    builder.add_edge(DECLINE_NODE, UPDATE_PROFILE_NODE)
    builder.add_edge(RECOMMENDER_NODE, UPDATE_PROFILE_NODE)
    builder.add_edge(UPDATE_PROFILE_NODE, END)

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------


def _iter_tool_events(update: dict[str, Any]) -> Iterator[tuple[str, str, dict[str, Any], Any]]:
    """Yield ``("tool", name, args, observation)`` events from one update.

    LangGraph's ``stream_mode="updates"`` yields ``{node_name: state_delta}``
    payloads. We inspect the messages emitted by each node, pair AIMessage
    tool-call requests with their following ToolMessage observations, and
    yield a flat tuple per tool invocation. When a tool call has no
    matching observation in the same delta (e.g. the observation lands in
    a later super-step), ``observation`` is ``None`` -- the consumer can
    still display the call.
    """

    # ToolMessage is imported lazily because it is only needed when the
    # graph actually emits tool events.
    from langchain_core.messages import AIMessage as _AIMessage
    from langchain_core.messages import ToolMessage as _ToolMessage

    # Pool tool messages from every node in this update so call/observation
    # pairing works regardless of which node emitted what.
    tool_messages: dict[str, _ToolMessage] = {}
    ai_messages: list[_AIMessage] = []
    for node_delta in update.values():
        if not isinstance(node_delta, dict):
            continue
        for msg in node_delta.get("messages", []) or []:
            if isinstance(msg, _ToolMessage):
                tool_call_id = getattr(msg, "tool_call_id", None)
                if tool_call_id:
                    tool_messages[tool_call_id] = msg
            elif isinstance(msg, _AIMessage):
                ai_messages.append(msg)

    for ai in ai_messages:
        for tool_call in getattr(ai, "tool_calls", None) or []:
            if isinstance(tool_call, dict):
                name = str(tool_call.get("name", ""))
                args = tool_call.get("args") or {}
                tool_call_id = tool_call.get("id")
            else:
                name = str(getattr(tool_call, "name", ""))
                args = getattr(tool_call, "args", {}) or {}
                tool_call_id = getattr(tool_call, "id", None)
            observation: Any = None
            if tool_call_id and tool_call_id in tool_messages:
                observation = tool_messages[tool_call_id].content
            yield ("tool", name, args if isinstance(args, dict) else {}, observation)


def _final_answer(update: dict[str, Any]) -> str | None:
    """Return the text of the last AIMessage with no pending tool calls.

    Returns ``None`` when this update does not contain a user-facing
    final answer (e.g. it is an intermediate tool-call AIMessage or a
    state-only update).
    """

    from langchain_core.messages import AIMessage as _AIMessage

    final_text: str | None = None
    for node_delta in update.values():
        if not isinstance(node_delta, dict):
            continue
        for msg in node_delta.get("messages", []) or []:
            if isinstance(msg, _AIMessage) and not (getattr(msg, "tool_calls", None) or []):
                content = msg.content
                if isinstance(content, str) and content:
                    final_text = content
    return final_text


def stream_graph(
    graph: CompiledStateGraph,
    input_state: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> Iterator[tuple[Any, ...]]:
    """Stream tool events and the final answer from a graph run.

    Yields a uniform sequence of:

    * ``("tool", tool_name, args, observation)`` -- one per tool call.
    * ``("final", content)`` -- emitted exactly once at the end with the
      text of the agent's final user-facing answer.

    The implementation uses ``graph.stream(..., stream_mode="updates")``
    so each ``update`` describes a single super-step. Tool events are
    yielded as they arrive, and the final answer is held back until the
    stream completes -- guaranteeing the ordering invariant required by
    Property 11 (every tool-call event index is strictly less than the
    final-answer event index).

    Args:
        graph: The compiled graph returned by :func:`build_graph`.
        input_state: Input passed to ``graph.stream`` (typically
            ``{"messages": [HumanMessage(content=user_query)]}``).
        config: LangGraph runnable config; should include
            ``{"configurable": {"thread_id": <session_id>, "user_id": <user_id>}}``.

    Yields:
        Tuples as described above.
    """

    final_content: str | None = None
    for update in graph.stream(
        input_state,
        config=config,
        stream_mode="updates",
    ):
        if not isinstance(update, dict):
            continue
        for event in _iter_tool_events(update):
            yield event
        candidate = _final_answer(update)
        if candidate is not None:
            final_content = candidate

    yield ("final", final_content if final_content is not None else "")


__all__ = [
    "CONFIRMATION_NODE",
    "DECLINE_NODE",
    "LOAD_PROFILE_NODE",
    "QUERY_ROUTER_NODE",
    "REACT_AGENT_NODE",
    "RECOMMENDER_NODE",
    "SUMMARIZE_NODE",
    "UPDATE_PROFILE_NODE",
    "build_graph",
    "stream_graph",
]
