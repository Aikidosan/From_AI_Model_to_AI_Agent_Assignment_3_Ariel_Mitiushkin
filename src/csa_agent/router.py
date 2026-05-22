"""Query Router for the Customer Service Data Analyst Agent.

The Query Router is a single LLM-backed classifier that labels each
incoming user query as one of three :class:`RouteLabel` values:

* ``STRUCTURED`` — the query is answerable by filtering / counting the
  Bitext customer service dataset (e.g. *"How many refund requests are
  there?"*). The Agent routes these to the ReAct tool-calling path.
* ``UNSTRUCTURED`` — the query asks for a narrative or summary grounded
  in the dataset (e.g. *"Summarize the FEEDBACK category"*). The Agent
  routes these to the summarization path.
* ``OUT_OF_SCOPE`` — anything else (general world knowledge, opinions,
  topics unrelated to the Bitext dataset). The Agent declines politely
  without invoking the LLM's general knowledge.

The classification prompt is intentionally narrow and the call uses
**no tool bindings** so the router never executes dataset tools during
classification — satisfying Requirement 2.5.

This function is **total**: any LLM exception, malformed response, or
unrecognised label is coerced to :attr:`RouteLabel.OUT_OF_SCOPE`. This
gives Property 7 (well-formed router output) for free and lets the
caller treat the result as a plain enum without defensive checks.

Validates:

* Requirement 2.1 — every query is classified into exactly one of the
  three labels before any tool is invoked.
* Requirement 2.5 — classification uses a focused LLM prompt with no
  tool bindings.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from langchain_openai import ChatOpenAI


class RouteLabel(str, Enum):
    """The three routing labels produced by :func:`classify_query`."""

    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"
    OUT_OF_SCOPE = "out_of_scope"


class _RouterDecision(BaseModel):
    """Structured-output schema wrapping a single :class:`RouteLabel`.

    LangChain's ``with_structured_output`` is most reliable when given a
    Pydantic model rather than a bare ``str``-Enum, so we wrap the label
    in a small model and unwrap it after the call.
    """

    label: RouteLabel = Field(
        ...,
        description=(
            "Routing label for the user query. Must be one of "
            "'structured', 'unstructured', or 'out_of_scope'."
        ),
    )


_SYSTEM_PROMPT = (
    "You are the query router for a Customer Service Data Analyst Agent that "
    "answers questions about the Bitext Customer Service Tagged Training "
    "Dataset (a CSV of customer utterances labelled with category and intent, "
    "e.g. REFUND/track_refund, ORDER/cancel_order, FEEDBACK/...).\n\n"
    "Classify the user's query into exactly one of three labels:\n"
    "  - 'structured': the query can be answered by filtering or counting "
    "rows in the dataset (e.g. 'How many refund requests are there?', "
    "'List the categories', 'Show 5 examples of cancel_order').\n"
    "  - 'unstructured': the query asks for a narrative, description, or "
    "summary grounded in the dataset (e.g. 'Summarize the FEEDBACK "
    "category', 'Describe what kinds of complaints customers submit').\n"
    "  - 'out_of_scope': anything else, including general world knowledge, "
    "opinions, coding help, or topics unrelated to the Bitext customer "
    "service dataset (e.g. 'What is the capital of France?', 'Write me a "
    "poem', 'How do I cook pasta?').\n\n"
    "Output strictly one of the three labels. Do not explain your choice."
)


def _coerce_label(raw: object) -> RouteLabel:
    """Best-effort coercion of an arbitrary value to a :class:`RouteLabel`.

    Accepts ``RouteLabel`` instances, the underlying string values, and
    case-insensitive variants (``"Structured"``, ``"OUT_OF_SCOPE"``, ...).
    Anything else falls through to :attr:`RouteLabel.OUT_OF_SCOPE` so the
    caller never has to handle ``None`` or unexpected strings.
    """

    if isinstance(raw, RouteLabel):
        return raw
    if isinstance(raw, str):
        try:
            return RouteLabel(raw.strip().lower())
        except ValueError:
            return RouteLabel.OUT_OF_SCOPE
    return RouteLabel.OUT_OF_SCOPE


def classify_query(user_query: str, llm: "ChatOpenAI") -> RouteLabel:
    """Classify ``user_query`` into one of three :class:`RouteLabel` values.

    A single LLM call is made with a focused classification prompt and
    **no tool bindings** (Requirement 2.5). The LLM is wrapped with
    ``with_structured_output`` so it returns a :class:`_RouterDecision`
    whose ``label`` field is a :class:`RouteLabel`.

    The function is total: any exception (network error, parse failure,
    unrecognised label) is caught and mapped to
    :attr:`RouteLabel.OUT_OF_SCOPE`. This is the safe default — declining
    a query the agent cannot confidently route is better than falling
    through to general-knowledge answers (Requirement 2.2).

    Parameters
    ----------
    user_query:
        The latest user message text.
    llm:
        A pre-constructed :class:`~langchain_openai.ChatOpenAI` instance,
        typically obtained from :func:`csa_agent.llm.get_llm`. The
        function never calls ``get_llm()`` itself, so tests can pass a
        fake or mock LLM here.

    Returns
    -------
    RouteLabel
        Always one of :attr:`RouteLabel.STRUCTURED`,
        :attr:`RouteLabel.UNSTRUCTURED`, or :attr:`RouteLabel.OUT_OF_SCOPE`.
    """

    try:
        structured_llm = llm.with_structured_output(_RouterDecision)
        response = structured_llm.invoke(
            [
                ("system", _SYSTEM_PROMPT),
                ("human", user_query),
            ]
        )
    except Exception:  # noqa: BLE001 - any failure must yield a safe default
        return RouteLabel.OUT_OF_SCOPE

    # ``with_structured_output`` normally returns the Pydantic model
    # directly, but some providers / versions return a dict. Handle both.
    if isinstance(response, _RouterDecision):
        return _coerce_label(response.label)
    if isinstance(response, dict):
        return _coerce_label(response.get("label"))
    return _coerce_label(response)


__all__ = ["RouteLabel", "classify_query"]
