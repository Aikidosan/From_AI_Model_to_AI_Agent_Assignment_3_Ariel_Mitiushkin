"""Property test for the user profile store.

Feature: customer-service-data-analyst-agent
Property 12: Profile round-trip and topic counter.

Validates: Requirements 7.4, 7.5.

Two sub-properties are encoded here:

* **Round-trip.** For any :class:`UserProfile` ``p``,
  ``load_profile(save_profile(p).user_id) == p`` (compared on the
  semantically meaningful fields). ``save_profile`` refreshes
  ``updated_at`` to the current UTC time, so we cannot use full
  Pydantic equality directly: the loaded profile carries the
  saved-with-updated-at version, while the generated input still has
  its original ``updated_at``. We compare the durable fields
  explicitly: ``user_id``, ``name``, ``sorted(frequent_topics)``,
  ``preferences``, and ``topic_counts``.

* **Counter iff invariant.** For any sequence of
  ``record_topic(profile, topic, repeat=k)`` calls,
  ``t in profile.frequent_topics`` if and only if
  ``profile.topic_counts[t] >= 3`` (where the threshold is
  ``FREQUENT_TOPIC_THRESHOLD``). Each property example starts from a
  fresh profile so the iff holds without depending on whatever
  inconsistent counters Hypothesis might generate.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from csa_agent.profile import (
    FREQUENT_TOPIC_THRESHOLD,
    UserProfile,
    load_profile,
    record_topic,
    save_profile,
)

from tests.strategies import user_id_st, user_profiles_st


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _semantic_key(profile: UserProfile) -> tuple:
    """Return the comparable, durable fields of ``profile``.

    Equality on the full :class:`UserProfile` model is too strict
    because ``save_profile`` always refreshes ``updated_at`` to the
    current UTC time. The fields below are the ones that survive a
    round-trip unchanged:

    * ``user_id`` -- the storage key, preserved exactly.
    * ``name`` -- free-form, preserved exactly.
    * ``frequent_topics`` -- compared as a sorted list because order
      is not part of the semantic contract (the design only requires
      that promoted topics are present in the list).
    * ``preferences`` -- preserved exactly.
    * ``topic_counts`` -- preserved exactly.
    """

    return (
        profile.user_id,
        profile.name,
        sorted(profile.frequent_topics),
        dict(profile.preferences),
        dict(profile.topic_counts),
    )


# ---------------------------------------------------------------------------
# Sub-property 1: save_profile then load_profile preserves the durable fields
# ---------------------------------------------------------------------------


@given(profile=user_profiles_st())
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_save_then_load_profile_round_trip(tmp_path_factory, profile: UserProfile) -> None:
    """Feature: customer-service-data-analyst-agent, Property 12: Profile round-trip and topic counter.

    Validates Requirements 7.5.

    Saving a generated profile and loading it back yields a profile
    whose durable fields equal the input. We use a fresh profile
    directory per example via ``tmp_path_factory`` so test cases never
    collide on the same ``user_id``.
    """

    profile_dir = tmp_path_factory.mktemp("profiles")

    saved = save_profile(profile, profile_dir=str(profile_dir))
    loaded = load_profile(saved.user_id, profile_dir=str(profile_dir))

    # The loaded profile is what is on disk, which equals the saved
    # version (with refreshed ``updated_at``). Semantic equality is the
    # contract we care about.
    assert _semantic_key(saved) == _semantic_key(loaded)
    assert _semantic_key(profile) == _semantic_key(loaded)


# ---------------------------------------------------------------------------
# Sub-property 2: record_topic establishes the iff invariant
# ---------------------------------------------------------------------------


# A topic event is a (topic, repeat_count) pair: call ``record_topic`` for
# the same topic ``repeat_count`` times. We constrain ``repeat_count`` to
# 1..6 so even modest sequences cross the promotion threshold of 3.
_TOPIC_ALPHABET = ("alpha", "beta", "gamma", "delta", "epsilon")


_topic_event_st = st.tuples(
    st.sampled_from(_TOPIC_ALPHABET),
    st.integers(min_value=1, max_value=6),
)


@given(user_id=user_id_st(), events=st.lists(_topic_event_st, min_size=0, max_size=12))
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_record_topic_satisfies_frequent_topic_iff_invariant(
    user_id: str, events: list[tuple[str, int]]
) -> None:
    """Feature: customer-service-data-analyst-agent, Property 12: Profile round-trip and topic counter.

    Validates Requirements 7.4.

    Driving any sequence of ``record_topic`` calls from a fresh profile,
    a topic must appear in ``frequent_topics`` if and only if its
    counter is at least :data:`FREQUENT_TOPIC_THRESHOLD`.
    """

    # Start from a clean profile so the iff cannot be poisoned by
    # generator-supplied counters that are inconsistent on construction.
    profile = UserProfile(user_id=user_id)

    for topic, repeat in events:
        for _ in range(repeat):
            record_topic(profile, topic)

    # Forward direction: every topic listed as frequent must have a
    # count at or above the threshold.
    for topic in profile.frequent_topics:
        assert profile.topic_counts.get(topic, 0) >= FREQUENT_TOPIC_THRESHOLD, (
            f"{topic!r} is in frequent_topics but its count is "
            f"{profile.topic_counts.get(topic, 0)}"
        )

    # Reverse direction: every topic whose count meets the threshold
    # must appear in the frequent_topics list.
    for topic, count in profile.topic_counts.items():
        if count >= FREQUENT_TOPIC_THRESHOLD:
            assert topic in profile.frequent_topics, (
                f"{topic!r} has count {count} >= threshold but is not in "
                f"frequent_topics"
            )

    # Idempotence: a topic must never appear twice in frequent_topics
    # even if record_topic is called many more times after promotion.
    assert len(profile.frequent_topics) == len(set(profile.frequent_topics)), (
        f"frequent_topics contains duplicates: {profile.frequent_topics!r}"
    )
