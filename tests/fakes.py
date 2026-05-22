"""Test doubles used in lieu of a real Nebius LLM client.

These fakes implement the smallest subset of the :class:`ChatOpenAI`
surface area that the agent code actually touches, so tests can exercise
control flow, structured-output coercion, tool-calling, and recursion
caps without ever hitting the network. They are deliberately *not*
LangChain ``BaseChatModel`` subclasses because the agent only uses
``invoke``, ``with_structured_output``, ``bind_tools``, ``stream``, and
the ``base_url`` attribute — and a tiny duck-typed object is much
easier to reason about in tests than a fully-mocked LangChain model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable


# ---------------------------------------------------------------------------
# Structured-output fake (used by the router)
# ---------------------------------------------------------------------------


@dataclass
class FakeStructuredLLM:
    """Returns a pre-set payload from ``invoke``, ignores the input.

    ``payload`` may be a Pydantic model instance, a ``dict``, a string,
    or anything else — the router's ``_coerce_label`` will normalise it.
    """

    payload: Any
    invocations: list[Any] = field(default_factory=list)

    def invoke(self, messages: Any, *_args: Any, **_kwargs: Any) -> Any:
        self.invocations.append(messages)
        return self.payload


@dataclass
class FakeChatModel:
    """Minimal stand-in for ``ChatOpenAI`` used in router/recommender tests.

    Construct with the payload that ``with_structured_output(...).invoke``
    should return. ``raise_on_call`` is an optional exception class that
    will be raised from ``invoke`` to exercise the router's "any failure
    -> OUT_OF_SCOPE" branch.
    """

    structured_payload: Any = None
    raise_on_call: type[BaseException] | None = None
    invoke_response: Any = None  # Used by recommender / generic invoke.
    invocations: list[Any] = field(default_factory=list)
    base_url: str = "https://api.studio.nebius.ai/v1/"  # Property-14 marker.

    # ------------------------------------------------------------------ #
    # Surface used by router.classify_query
    # ------------------------------------------------------------------ #
    def with_structured_output(self, _model: Any) -> FakeStructuredLLM:
        if self.raise_on_call is not None:
            raise self.raise_on_call("simulated failure")
        return FakeStructuredLLM(payload=self.structured_payload)

    # ------------------------------------------------------------------ #
    # Surface used by recommender.generate_suggestions
    # ------------------------------------------------------------------ #
    def invoke(self, messages: Any, *_args: Any, **_kwargs: Any) -> Any:
        self.invocations.append(messages)
        if self.raise_on_call is not None:
            raise self.raise_on_call("simulated failure")
        return self.invoke_response


# ---------------------------------------------------------------------------
# ReAct fake (used by graph / streaming / iteration-cap tests)
# ---------------------------------------------------------------------------


@dataclass
class _ScriptedToolCall:
    """A pre-scripted tool call for :class:`ScriptedReActModel`."""

    name: str
    args: dict[str, Any]
    id: str


class ScriptedReActModel(Runnable):
    """Fake chat model that emits a scripted sequence of tool calls then a final answer.

    Lifecycle:

    1. The first ``len(scripted)`` invokes return AIMessages whose
       ``tool_calls`` request the next scripted tool.
    2. After all scripted tools have been called, subsequent invokes
       return a plain AIMessage carrying ``final_text`` (no tool calls).

    Used by ReAct integration tests to assert ordering, iteration cap,
    and streaming semantics without hitting Nebius.

    Subclasses :class:`langchain_core.runnables.Runnable` so it can be
    composed with prompts (``prompt | model``) the way LangGraph's
    :func:`create_react_agent` builds its inner runnable.
    """

    def __init__(
        self,
        scripted: Iterable[tuple[str, dict[str, Any]]] | None = None,
        final_text: str = "Done.",
        always_tool_call: bool = False,
        always_tool_name: str = "list_categories",
    ) -> None:
        self._calls: list[_ScriptedToolCall] = [
            _ScriptedToolCall(name=n, args=a, id=f"call_{i}")
            for i, (n, a) in enumerate(scripted or [])
        ]
        self._final_text = final_text
        self._always_tool_call = always_tool_call
        self._always_tool_name = always_tool_name
        self._cursor = 0
        self.invocations: list[Any] = []
        self.base_url = "https://api.studio.nebius.ai/v1/"

    def bind_tools(self, _tools: Any, **_kw: Any) -> "ScriptedReActModel":
        # ReAct binds tools onto the model; just return self so chained
        # calls keep returning the scripted behaviour.
        return self

    def bind(self, **_kw: Any) -> "ScriptedReActModel":
        return self

    def with_config(self, *_a: Any, **_kw: Any) -> "ScriptedReActModel":
        return self

    def invoke(self, messages: Any, *_a: Any, **_kw: Any) -> AIMessage:
        self.invocations.append(messages)

        if self._always_tool_call:
            tc = {
                "name": self._always_tool_name,
                "args": {},
                "id": f"call_inf_{len(self.invocations)}",
                "type": "tool_call",
            }
            return AIMessage(content="", tool_calls=[tc])

        if self._cursor < len(self._calls):
            sc = self._calls[self._cursor]
            self._cursor += 1
            tc = {
                "name": sc.name,
                "args": sc.args,
                "id": sc.id,
                "type": "tool_call",
            }
            return AIMessage(content="", tool_calls=[tc])

        return AIMessage(content=self._final_text)


# ---------------------------------------------------------------------------
# Helper: a callable LLM factory patch that returns the same instance
# ---------------------------------------------------------------------------


def llm_factory_returning(model: Any) -> Callable[..., Any]:
    """Return a callable that always returns ``model``.

    Useful as a ``monkeypatch.setattr(csa_agent.llm, 'get_llm', ...)``
    target so every site that calls ``get_llm()`` gets the same fake
    instance.
    """

    def _factory(*_a: Any, **_kw: Any) -> Any:
        return model

    return _factory


__all__ = [
    "FakeChatModel",
    "FakeStructuredLLM",
    "ScriptedReActModel",
    "llm_factory_returning",
]
