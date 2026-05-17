"""Per-user profile store for the Customer Service Data Analyst Agent.

Profiles capture lightweight personalization data (name, frequent topics,
free-form preferences) that survives across CLI invocations and conversation
sessions. Storage is intentionally separate from the LangGraph checkpointer
so a user's profile is preserved even when their conversation thread is
deleted.

Each profile lives at ``{profile_dir}/{user_id}.json`` where ``profile_dir``
defaults to :data:`Settings.profile_dir`. Writes are performed atomically
via a temp file plus :func:`os.replace`, eliminating partial-write hazards
if the process is killed mid-save (Requirement 7.5).

Public API:

- :class:`UserProfile` -- the Pydantic model persisted on disk.
- :func:`load_profile` -- read a profile by ``user_id`` or return a fresh one.
- :func:`save_profile` -- atomically persist a profile, refreshing
  ``updated_at`` and creating ``profile_dir`` if necessary.
- :func:`record_topic` -- increment a topic counter and promote the topic
  into ``frequent_topics`` when its count reaches 3 (Requirement 7.4).
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from typing import Final

from pydantic import BaseModel, Field

from .config import get_settings


# Threshold at which a topic is considered "frequent" and added to
# ``UserProfile.frequent_topics`` (Requirement 7.4).
FREQUENT_TOPIC_THRESHOLD: Final[int] = 3


class UserProfile(BaseModel):
    """Persistent per-user profile.

    Fields mirror the design's "User Profile" section. ``topic_counts`` is
    an internal counter used by :func:`record_topic` to decide when a topic
    should be promoted into :attr:`frequent_topics`; it is persisted so
    counters survive process restarts.
    """

    user_id: str
    name: str | None = None
    frequent_topics: list[str] = Field(default_factory=list)  # Req 7.1, 7.4
    preferences: dict[str, str] = Field(default_factory=dict)  # Req 7.1
    topic_counts: dict[str, int] = Field(default_factory=dict)  # internal counter
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


def _resolve_profile_dir(profile_dir: str | None) -> str:
    """Return the directory used for profile JSON files.

    Falls back to :func:`config.get_settings` when ``profile_dir`` is not
    explicitly provided so callers do not have to thread ``Settings`` through
    every call site.
    """

    return profile_dir if profile_dir is not None else get_settings().profile_dir


def _profile_path(user_id: str, profile_dir: str | None = None) -> str:
    """Return the on-disk path for ``user_id``'s profile JSON file."""

    return os.path.join(_resolve_profile_dir(profile_dir), f"{user_id}.json")


def load_profile(user_id: str, profile_dir: str | None = None) -> UserProfile:
    """Load the profile for ``user_id`` or return a fresh one if absent.

    A missing file is not an error: the agent should be usable on first run
    for any user identifier (including the default ``"default"`` user).
    A corrupted file is also treated as "no profile yet" rather than
    propagating a parse error mid-conversation; this matches the design's
    "profile saves are best-effort and never block the main response"
    posture.

    Args:
        user_id: Identifier used by the CLI's ``--user`` flag.
        profile_dir: Optional override for the profile directory; defaults
            to ``Settings.profile_dir``.

    Returns:
        Either the :class:`UserProfile` parsed from disk or a freshly
        constructed one with timestamps set to "now".
    """

    path = _profile_path(user_id, profile_dir)

    if not os.path.isfile(path):
        return UserProfile(user_id=user_id)

    try:
        with open(path, encoding="utf-8") as fp:
            raw = fp.read()
        profile = UserProfile.model_validate_json(raw)
    except (OSError, ValueError):
        # Treat unreadable / malformed profiles as "start fresh" so a
        # corrupt file does not block the user from getting answers.
        return UserProfile(user_id=user_id)

    # Defensive: ensure the loaded profile's user_id matches the requested
    # one. If a user manually moved files around, prefer the requested id
    # over whatever happens to be in the file.
    if profile.user_id != user_id:
        profile = profile.model_copy(update={"user_id": user_id})
    return profile


def save_profile(profile: UserProfile, profile_dir: str | None = None) -> UserProfile:
    """Atomically persist ``profile`` to disk and return the saved copy.

    Behavior:

    * Creates the profile directory if it does not yet exist (Requirement 7.5).
    * Refreshes :attr:`UserProfile.updated_at` to the current UTC time so
      callers always see a fresh modification timestamp.
    * Writes to a temp file in the same directory, then uses
      :func:`os.replace` to swap it into place. ``os.replace`` is atomic on
      POSIX and Windows for files on the same filesystem, so a crash mid
      write leaves either the previous file or the new file intact -- never
      a half-written one.

    Args:
        profile: The profile to persist. The returned instance has the
            updated timestamp; the input is not mutated in place.
        profile_dir: Optional override for the profile directory; defaults
            to ``Settings.profile_dir``.

    Returns:
        The saved :class:`UserProfile` (a copy with refreshed ``updated_at``).
    """

    directory = _resolve_profile_dir(profile_dir)
    os.makedirs(directory, exist_ok=True)

    refreshed = profile.model_copy(update={"updated_at": datetime.utcnow()})
    payload = refreshed.model_dump_json(indent=2)

    final_path = os.path.join(directory, f"{refreshed.user_id}.json")

    # Use a NamedTemporaryFile in the same directory so os.replace stays on
    # one filesystem (and therefore atomic). delete=False because we hand
    # the temp path to os.replace ourselves.
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{refreshed.user_id}.",
        suffix=".json.tmp",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            fp.write(payload)
        os.replace(tmp_path, final_path)
    except Exception:
        # Best-effort cleanup of the temp file on failure; re-raise so the
        # caller can decide whether to surface the error.
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise

    return refreshed


def record_topic(profile: UserProfile, topic: str) -> UserProfile:
    """Increment the count for ``topic`` and promote it when frequent.

    Mutates ``profile`` in place and also returns it, so callers can use
    either the mutated argument or the returned value interchangeably.

    Promotion rules (Requirement 7.4):

    * Increment ``topic_counts[topic]`` by 1.
    * If the new count is at least :data:`FREQUENT_TOPIC_THRESHOLD` (3) and
      ``topic`` is not already in ``frequent_topics``, append it.
    * Calls beyond the threshold are idempotent: the topic is never
      duplicated in ``frequent_topics``.

    Args:
        profile: The profile to update.
        topic: The category or intent name observed in the current turn.

    Returns:
        The same ``profile`` instance, mutated.
    """

    new_count = profile.topic_counts.get(topic, 0) + 1
    profile.topic_counts[topic] = new_count

    if new_count >= FREQUENT_TOPIC_THRESHOLD and topic not in profile.frequent_topics:
        profile.frequent_topics.append(topic)

    return profile


__all__ = [
    "FREQUENT_TOPIC_THRESHOLD",
    "UserProfile",
    "load_profile",
    "record_topic",
    "save_profile",
]
