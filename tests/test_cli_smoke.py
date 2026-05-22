"""CLI smoke tests for ``main.py``.

These tests spawn ``python main.py`` as a real subprocess with scripted
stdin and assert the contracts pinned by Requirement 5:

* 5.1 -- ``python main.py`` enters the interactive loop.
* 5.2 -- ``--session`` is accepted.
* 5.3 -- when ``--session`` is omitted, a fresh session ID is generated
  and displayed.
* 5.5 -- ``exit`` and ``quit`` terminate the loop gracefully (process
  returns 0).
* 5.6 -- ``--user`` is accepted.

The tests deliberately do *not* exercise a model turn -- they just
confirm the loop boots, recognises the args, and exits cleanly. A stub
``NEBIUS_API_KEY`` is supplied so :func:`csa_agent.config.get_settings`
does not exit at startup; no network call is made because the user
sends ``exit``/``quit`` immediately.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
_MAIN_PY = _REPO_ROOT / "main.py"

# UUIDv4 string as printed by ``str(uuid.uuid4())`` -- 8-4-4-4-12 hex.
_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_SESSION_LINE_RE = re.compile(r"\[Session:\s*(" + _UUID_RE.pattern + r")\]")

# Generous but bounded; the CLI loads the bundled CSV, builds the
# ReAct sub-agent, and opens a SqliteSaver before reading stdin, so
# startup commonly takes 10-15s on a developer laptop. 45s is
# comfortably above that even on a slow CI runner under load.
_PROC_TIMEOUT_SECONDS = 45.0


def _build_env() -> dict[str, str]:
    """Build the environment for the CLI subprocess.

    * ``NEBIUS_API_KEY`` is stubbed so ``get_settings`` does not exit.
      No real network call happens because the test always sends
      ``exit``/``quit`` before triggering a turn.
    * ``PYTHONPATH`` points at ``src/`` so ``import csa_agent`` works
      without an editable install (matches how ``conftest.py`` rigs the
      path for in-process tests).
    * ``PYTHONIOENCODING=utf-8`` forces UTF-8 on Windows so the tool
      glyph in main.py can be encoded if it ever appears.
    * ``PYTHONUNBUFFERED=1`` so stdout flushes promptly on exit.
    """

    env = os.environ.copy()
    env["NEBIUS_API_KEY"] = "stub-key"
    existing_pp = env.get("PYTHONPATH", "")
    parts = [str(_SRC_DIR)]
    if existing_pp:
        parts.append(existing_pp)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


@pytest.fixture
def run_cli():
    """Return a helper that spawns ``main.py`` with scripted stdin.

    The helper takes ``args`` (extra CLI args) and ``stdin_text``,
    spawns the process, sends ``stdin_text``, captures stdout/stderr,
    and returns ``(returncode, stdout, stderr)``. If the process hangs
    past :data:`_PROC_TIMEOUT_SECONDS`, it is killed and the test fails
    loudly so a regression in the exit handling cannot silently turn
    into a flaky timeout.
    """

    def _run(
        *,
        args: list[str] | None = None,
        stdin_text: str,
    ) -> tuple[int, str, str]:
        cmd: list[str] = [sys.executable, str(_MAIN_PY)]
        if args:
            cmd.extend(args)

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(_REPO_ROOT),
            env=_build_env(),
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            stdout, stderr = proc.communicate(
                input=stdin_text,
                timeout=_PROC_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            # Drain whatever is left so the file descriptors close.
            try:
                stdout, stderr = proc.communicate(timeout=2.0)
            except Exception:
                stdout, stderr = "", ""
            pytest.fail(
                "CLI process did not exit within "
                f"{_PROC_TIMEOUT_SECONDS}s. "
                f"Partial stdout={stdout!r} stderr={stderr!r}"
            )

        return proc.returncode, stdout, stderr

    return _run


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cli_prints_generated_session_id_when_session_omitted(run_cli):
    """Validates: Requirements 5.1, 5.3, 5.5.

    When ``--session`` is omitted, ``main.py`` must print
    ``[Session: <uuid>]`` and then enter the loop. Sending ``exit``
    terminates the loop with a clean exit code.
    """

    returncode, stdout, stderr = run_cli(stdin_text="exit\n")

    assert returncode == 0, (
        f"CLI did not exit cleanly. stdout={stdout!r} stderr={stderr!r}"
    )
    match = _SESSION_LINE_RE.search(stdout)
    assert match, (
        "Expected '[Session: <uuid>]' in stdout when --session is omitted. "
        f"Got stdout={stdout!r}"
    )

    # The captured group must be a real, parseable UUID -- not just
    # something matching a permissive regex.
    session_id = match.group(1)
    parsed = uuid.UUID(session_id)
    assert str(parsed) == session_id.lower(), (
        f"Generated session ID is not a canonical UUID: {session_id!r}"
    )


def test_cli_accepts_session_and_user_args(run_cli):
    """Validates: Requirements 5.1, 5.2, 5.5, 5.6.

    Passing ``--session`` and ``--user`` must be accepted without an
    argparse error, and sending ``exit`` must terminate cleanly.
    """

    returncode, stdout, stderr = run_cli(
        args=["--session", "my-test-session", "--user", "alice"],
        stdin_text="exit\n",
    )

    assert returncode == 0, (
        f"CLI did not exit cleanly with --session/--user. "
        f"stdout={stdout!r} stderr={stderr!r}"
    )
    # When --session is supplied explicitly, the CLI must NOT generate
    # and print a fresh ID -- the user already knows theirs.
    assert "[Session:" not in stdout, (
        "Expected no generated [Session: ...] line when --session is "
        f"provided. Got stdout={stdout!r}"
    )


def test_cli_exits_cleanly_on_quit(run_cli):
    """Validates: Requirements 5.1, 5.5.

    Both ``exit`` and ``quit`` must terminate the loop gracefully. The
    other tests already cover ``exit``; this one pins ``quit``.
    """

    returncode, stdout, stderr = run_cli(stdin_text="quit\n")

    assert returncode == 0, (
        f"CLI did not exit cleanly on 'quit'. "
        f"stdout={stdout!r} stderr={stderr!r}"
    )
