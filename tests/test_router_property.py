"""Property test for the Query Router classifier.

Feature: customer-service-data-analyst-agent
Property 7: Query Router output is well-formed.

Validates: Requirements 2.1.

For *any* user query string and *for any* raw value the LLM might
return from ``with_structured_output(...).invoke(...)``, the
:func:`csa_agent.router.classify_query` function returns a value that
is a member of :class:`~csa_agent.router.RouteLabel`.

The router is intentionally total: any LLM exception, malformed
response, or unrecognised label is coerced to
:attr:`RouteLabel.OUT_OF_SCOPE`. We exercise that contract here by
patching the LLM with arbitrary Hypothesis-generated payloads and by
forcing the structured wrapper to raise.

The LLM is replaced with a small fake whose
``with_structured_output(...).invoke(...)`` either returns a generated
payload (str, dict, int, ``None``, a valid :class:`RouteLabel`, or an
arbitrary :class:`pydantic.BaseModel` instance) or raises a configured
exception. No real network call is made.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from csa_agent.router import RouteLabel, classify_query


# ---------------------------------------------------------------------------
# LLM fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeStructured:
    """Inner double returned by ``with_structured_output``.

    ``invoke`` returns the pre-set ``payload`` and records the messages so
    the test can assert (if needed) that the router actually called us.
    """

    payload: Any
    invocations: list[Any]

    def invoke(self, messages: Any, *_a: Any, **_kw: Any) -> Any:
        self.invocations.append(messages)
        return self.payload


@dataclass
class _PayloadFakeChatModel:
    """Fake ChatOpenAI whose structured wrapper returns a generated payload.

    Constructed with the value that ``with_structured_output(...).invoke``
    should return. The fake mirrors the shape of
    :class:`tests.fakes.FakeChatModel` but specialises it for this
    property test so we can also generate arbitrary Pydantic models as
    payloads.
    """

    payload: Any
    invocations: list[Any]

    def with_structured_output(self, _schema: Any) -> _FakeStructured:
        return _FakeStructured(payload=self.payload, invocations=self.invocations)


@dataclass
class _RaisingFakeChatModel:
    """Fake ChatOpenAI whose ``with_structured_output`` raises.

    Exercises the router's "any failure -> OUT_OF_SCOPE" branch
    (Requirement 2.5 / Property 7's totality clause).
    """

    exc: BaseException

    def with_structured_output(self, _schema: Any) -> Any:
        raise self.exc


@dataclass
class _RaisingInvokeFakeChatModel:
    """Fake whose structured wrapper invokes-then-raises.

    Catches the ``invoke``-time exception path separately from the
    ``with_structured_output``-time path.
    """

    exc: BaseException

    def with_structured_output(self, _schema: Any) -> Any:
        return _RaisingInvoke(self.exc)


@dataclass
class _RaisingInvoke:
    exc: BaseException

    def invoke(self, *_a: Any, **_kw: Any) -> Any:
        raise self.exc


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


def _query_st() -> st.SearchStrategy[str]:
    """Strategy for arbitrary user query strings.

    The router treats the query as opaque text, so any unicode string
    (excluding surrogates) is fair game. Length is bounded to keep the
    example space tractable.
    """

    return st.text(
        alphabet=st.characters(blacklist_categories=("Cs",)),
        min_size=0,
        max_size=80,
    )


def _payload_st() -> st.SearchStrategy[Any]:
    """Strategy for arbitrary payloads returned from ``invoke``.

    Spans the universe the real LLM might produce (including via
    structured-output coercion glitches):

    * ``None``
    * Booleans, integers, floats
    * Arbitrary unicode strings -- including ones that happen to spell a
      ``RouteLabel`` value, plus garbage strings
    * Plain ``dict``s with or without a ``label`` key
    * Lists of small payloads
    * Genuine :class:`RouteLabel` enum members (the happy path)
    """

    label_strings = st.sampled_from(
        [label.value for label in RouteLabel]
        + ["STRUCTURED", "Out_Of_Scope", "  unstructured  ", "garbage", ""]
    )
    text_st = st.text(
        alphabet=st.characters(blacklist_categories=("Cs",)),
        min_size=0,
        max_size=30,
    )

    primitive_st = st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-100, max_value=100),
        st.floats(allow_nan=False, allow_infinity=False, width=32),
        text_st,
        label_strings,
        st.sampled_from(list(RouteLabel)),
    )

    dict_st = st.dictionaries(
        keys=st.sampled_from(["label", "route", "value", "other"]),
        values=st.one_of(label_strings, text_st, st.none(), st.integers()),
        max_size=4,
    )
    list_st = st.lists(primitive_st, max_size=3)

    return st.one_of(primitive_st, dict_st, list_st)


def _exception_st() -> st.SearchStrategy[BaseException]:
    """Strategy for exception instances thrown from inside the LLM call."""

    return st.sampled_from(
        [
            RuntimeError("simulated failure"),
            ValueError("bad parse"),
            TimeoutError("network timeout"),
            KeyError("missing field"),
        ]
    )


# ---------------------------------------------------------------------------
# Property 7 -- router output is always a RouteLabel member
# ---------------------------------------------------------------------------


@given(query=_query_st(), payload=_payload_st())
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_classify_query_returns_route_label_for_any_payload(
    query: str, payload: Any
) -> None:
    """Feature: customer-service-data-analyst-agent, Property 7: Query Router output is well-formed.

    Validates Requirements 2.1.

    For an arbitrary user query and an arbitrary raw payload returned
    from ``with_structured_output(...).invoke(...)``,
    :func:`classify_query` returns a :class:`RouteLabel` member. The
    router must never propagate ``None``, raw strings, dicts, or other
    junk to downstream callers.
    """

    fake = _PayloadFakeChatModel(payload=payload, invocations=[])
    result = classify_query(query, fake)  # type: ignore[arg-type]

    assert isinstance(result, RouteLabel), (
        f"classify_query returned {result!r} of type {type(result).__name__}; "
        f"expected a RouteLabel member"
    )


@given(query=_query_st(), exc=_exception_st())
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_classify_query_returns_out_of_scope_when_with_structured_output_raises(
    query: str, exc: BaseException
) -> None:
    """Feature: customer-service-data-analyst-agent, Property 7: Query Router output is well-formed.

    Validates Requirements 2.1.

    When ``with_structured_output`` itself raises (e.g. the LLM provider
    rejects the schema), the router must catch the exception and return
    :attr:`RouteLabel.OUT_OF_SCOPE` rather than propagating the failure.
    """

    fake = _RaisingFakeChatModel(exc=exc)
    result = classify_query(query, fake)  # type: ignore[arg-type]

    assert isinstance(result, RouteLabel)
    assert result is RouteLabel.OUT_OF_SCOPE, (
        f"expected OUT_OF_SCOPE on with_structured_output failure, got {result!r}"
    )


@given(query=_query_st(), exc=_exception_st())
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_classify_query_returns_out_of_scope_when_invoke_raises(
    query: str, exc: BaseException
) -> None:
    """Feature: customer-service-data-analyst-agent, Property 7: Query Router output is well-formed.

    Validates Requirements 2.1.

    When the structured wrapper's ``invoke`` raises (network error,
    timeout, validation failure on the LLM's response), the router
    must still return a :class:`RouteLabel` -- specifically the safe
    ``OUT_OF_SCOPE`` default.
    """

    fake = _RaisingInvokeFakeChatModel(exc=exc)
    result = classify_query(query, fake)  # type: ignore[arg-type]

    assert isinstance(result, RouteLabel)
    assert result is RouteLabel.OUT_OF_SCOPE
