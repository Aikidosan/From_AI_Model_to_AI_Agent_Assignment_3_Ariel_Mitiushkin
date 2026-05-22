"""Hypothesis strategies shared across property tests.

These strategies generate small, structurally correct DataFrames and
related primitives so individual property tests don't have to invent
their own. Alphabets are kept tiny on purpose so generated frames have
non-trivial overlap (filters return non-empty results in many cases),
which is more useful than maximally sparse data for verifying tool
contracts.
"""

from __future__ import annotations

from typing import Final

import pandas as pd
from hypothesis import strategies as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from csa_agent.profile import UserProfile


# Small alphabets keep examples short and ensure non-trivial overlap.
_CATEGORY_ALPHABET: Final[tuple[str, ...]] = (
    "REFUND",
    "ORDER",
    "FEEDBACK",
    "DELIVERY",
    "ACCOUNT",
)
_INTENT_ALPHABET: Final[tuple[str, ...]] = (
    "track_refund",
    "cancel_order",
    "rate_service",
    "delay_delivery",
    "create_account",
    "delete_account",
)


def category_st() -> st.SearchStrategy[str]:
    """Strategy for a single category drawn from the fixed alphabet."""

    return st.sampled_from(_CATEGORY_ALPHABET)


def intent_st() -> st.SearchStrategy[str]:
    """Strategy for a single intent drawn from the fixed alphabet."""

    return st.sampled_from(_INTENT_ALPHABET)


def utterance_st() -> st.SearchStrategy[str]:
    """Strategy for an utterance string of bounded length."""

    return st.text(
        alphabet=st.characters(blacklist_categories=("Cs", "Cc")),
        min_size=0,
        max_size=40,
    )


def dataframes_st(
    min_rows: int = 1,
    max_rows: int = 30,
) -> st.SearchStrategy[pd.DataFrame]:
    """Generate DataFrames with the columns required by every dataset tool.

    The frame always has ``utterance``, ``category``, ``intent`` columns;
    extra columns are not added because none of the tools rely on them.
    Row count is bounded so tests stay fast.
    """

    row_st = st.fixed_dictionaries(
        {
            "utterance": utterance_st(),
            "category": category_st(),
            "intent": intent_st(),
        }
    )

    return st.lists(row_st, min_size=min_rows, max_size=max_rows).map(
        lambda rows: pd.DataFrame(rows, columns=["utterance", "category", "intent"])
    )


def non_existent_string_st(df: pd.DataFrame) -> st.SearchStrategy[str]:
    """Strategy for a string guaranteed to be absent from ``df``.

    Achieved by prefixing the value with a sentinel that no alphabet in
    :func:`dataframes_st` can produce. We still avoid generating values
    that ``df`` happens to contain by composing a ``filter`` predicate.
    """

    present = set(df["category"].astype(str)) | set(df["intent"].astype(str))
    sentinel = "__SENTINEL_NOT_IN_DATASET__"

    return st.text(
        alphabet=st.characters(blacklist_categories=("Cs", "Cc")),
        min_size=0,
        max_size=20,
    ).map(lambda s: f"{sentinel}{s}").filter(lambda v: v not in present)


def n_examples_st() -> st.SearchStrategy[int]:
    """Strategy for ``show_examples.n`` within the validated 1..50 range."""

    return st.integers(min_value=1, max_value=50)


# ---------------------------------------------------------------------------
# Profile strategies (used by tests/test_profile_property.py)
# ---------------------------------------------------------------------------

# Keep the alphabets small so user_id, names, topics, and preference keys
# overlap across generated profiles. That gives Hypothesis a fair shot at
# exercising the "topic appears multiple times" branch in record_topic
# and the round-trip property without billowing the example space.
_USER_ID_ALPHABET: Final[str] = "abcdefghijklmnopqrstuvwxyz0123456789_-"
_TOPIC_ALPHABET: Final[tuple[str, ...]] = (
    "REFUND",
    "ORDER",
    "FEEDBACK",
    "DELIVERY",
    "ACCOUNT",
    "track_refund",
    "cancel_order",
    "rate_service",
    "delay_delivery",
)
_PREF_KEY_ALPHABET: Final[tuple[str, ...]] = (
    "tone",
    "language",
    "verbosity",
    "format",
)


def user_id_st() -> st.SearchStrategy[str]:
    """Strategy for filesystem-safe user identifiers.

    The character class avoids path separators, NUL bytes, and the
    Windows-reserved characters so generated ids never produce an
    OSError when used as a JSON filename component.
    """

    return st.text(alphabet=_USER_ID_ALPHABET, min_size=1, max_size=12)


def _topic_st() -> st.SearchStrategy[str]:
    """Strategy for a topic string drawn from the fixed alphabet.

    record_topic stores topics as plain strings keyed in the counter
    dict, so we keep the alphabet finite to encourage repeats (which
    is what triggers the frequent-topic promotion).
    """

    return st.sampled_from(_TOPIC_ALPHABET)


def user_profiles_st() -> st.SearchStrategy[UserProfile]:
    """Generate :class:`UserProfile` instances with realistic shapes.

    Every generated profile satisfies the model's invariants:

    * ``user_id`` is a non-empty filesystem-safe string.
    * ``name`` is either ``None`` or a short text value.
    * ``frequent_topics`` is a list of short strings drawn from the
      shared topic alphabet.
    * ``preferences`` maps short keys to short string values.
    * ``topic_counts`` is a dict whose values are all non-negative
      integers (the design's counter is a non-negative integer).

    The strategy intentionally allows ``frequent_topics`` and
    ``topic_counts`` to be inconsistent on construction: tests in
    ``test_profile_property.py`` either compare across a save/load
    round-trip (where consistency is preserved by serialization) or
    drive ``record_topic`` calls themselves to establish the iff
    invariant from a clean slate.
    """

    return st.builds(
        UserProfile,
        user_id=user_id_st(),
        name=st.one_of(
            st.none(),
            st.text(
                alphabet=st.characters(blacklist_categories=("Cs", "Cc")),
                min_size=0,
                max_size=20,
            ),
        ),
        frequent_topics=st.lists(_topic_st(), min_size=0, max_size=5, unique=True),
        preferences=st.dictionaries(
            keys=st.sampled_from(_PREF_KEY_ALPHABET),
            values=st.text(
                alphabet=st.characters(blacklist_categories=("Cs", "Cc")),
                min_size=0,
                max_size=15,
            ),
            max_size=4,
        ),
        topic_counts=st.dictionaries(
            keys=_topic_st(),
            values=st.integers(min_value=0, max_value=10),
            max_size=6,
        ),
    )


# ---------------------------------------------------------------------------
# Message strategies (used by tests/test_checkpointer_property.py)
# ---------------------------------------------------------------------------

# Tiny alphabets keep example payloads short. The checkpointer test only
# inspects the (type, content) tuple so the actual content text just has
# to round-trip through JSON-friendly serialization.
_MESSAGE_CONTENT_ALPHABET: Final[tuple[str, ...]] = (
    "hello",
    "world",
    "refund",
    "order",
    "ack",
    "ok",
    "follow up",
    "summary",
)
_TOOL_CALL_ID_ALPHABET: Final[str] = "abcdefghijklmnopqrstuvwxyz0123456789"


def _human_message_st() -> st.SearchStrategy[HumanMessage]:
    return st.builds(
        HumanMessage,
        content=st.sampled_from(_MESSAGE_CONTENT_ALPHABET),
    )


def _ai_message_st() -> st.SearchStrategy[AIMessage]:
    return st.builds(
        AIMessage,
        content=st.sampled_from(_MESSAGE_CONTENT_ALPHABET),
    )


def _tool_message_st() -> st.SearchStrategy[ToolMessage]:
    return st.builds(
        ToolMessage,
        content=st.sampled_from(_MESSAGE_CONTENT_ALPHABET),
        tool_call_id=st.text(
            alphabet=_TOOL_CALL_ID_ALPHABET, min_size=4, max_size=8
        ),
    )


def message_sequences_st(
    min_size: int = 1, max_size: int = 8
) -> st.SearchStrategy[list[BaseMessage]]:
    """Generate short lists of mixed Human/AI/Tool messages.

    The list is bounded (default 1..8) because the checkpointer test
    invokes the graph once per message, opening and closing the SQLite
    connection per Hypothesis example. Larger lists only slow the test
    down without expanding the state space meaningfully.
    """

    return st.lists(
        st.one_of(_human_message_st(), _ai_message_st(), _tool_message_st()),
        min_size=min_size,
        max_size=max_size,
    )


__all__ = [
    "category_st",
    "dataframes_st",
    "intent_st",
    "message_sequences_st",
    "n_examples_st",
    "non_existent_string_st",
    "user_id_st",
    "user_profiles_st",
    "utterance_st",
]
