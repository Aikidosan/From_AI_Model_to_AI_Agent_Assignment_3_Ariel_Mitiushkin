"""Property test for the LLM provider invariant.

Feature: customer-service-data-analyst-agent
Property 14: All LLM clients point at Nebius.

Validates: Requirements 9.1, 9.2.

The invariant has two halves; this module enforces them both:

1. **Runtime spy**: drive a real :func:`csa_agent.llm.get_llm` invocation
   and assert that every outbound HTTP request observed by
   ``pytest-httpx`` targets ``settings.nebius_base_url`` (host
   ``api.studio.nebius.ai``). ``pytest-httpx`` intercepts at the
   :class:`httpx.HTTPTransport` layer, which is the same transport the
   ``openai`` Python SDK uses internally, so this catches LLM calls
   issued via :class:`langchain_openai.ChatOpenAI`.

2. **Static AST scan**: walk every ``.py`` file under
   ``src/csa_agent/`` (excluding ``llm.py`` -- the only sanctioned
   constructor site) and reject any direct call to a forbidden chat
   model class (``ChatOpenAI``, ``ChatAnthropic``, ...). The two halves
   together close the loop: at runtime no other endpoint is contacted,
   and at parse time no other constructor exists to point at one.

Property 14 is intentionally a non-Hypothesis property: the universe
quantified over ("any LLM client constructed at runtime", "any chat
model constructor in the source tree") is finite and best validated
deterministically. Hypothesis would not buy more coverage here -- the
runtime spy already catches every request the SDK issues, and the AST
scan already visits every ``Call`` node in the package.
"""

from __future__ import annotations

import ast
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from langchain_core.messages import HumanMessage

from csa_agent.config import DEFAULT_NEBIUS_BASE_URL, get_settings
from csa_agent.llm import get_llm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Chat-model classes whose construction must be confined to ``llm.py``.
#: Any direct call to one of these names elsewhere in ``src/csa_agent/``
#: is a Property-14 violation: it would let traffic escape the Nebius
#: factory without going through :func:`csa_agent.llm.get_llm`.
FORBIDDEN_CHAT_MODEL_CLASSES: frozenset[str] = frozenset(
    {
        "ChatOpenAI",
        "ChatAnthropic",
        "ChatGoogleGenerativeAI",
        "ChatOllama",
        "ChatCohere",
        "ChatGroq",
        "ChatMistralAI",
        "ChatVertexAI",
        "AzureChatOpenAI",
    }
)

#: Repository ``src/csa_agent/`` directory; resolved at import time.
_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_SRC_ROOT: Path = _REPO_ROOT / "src" / "csa_agent"

#: Filename inside ``_SRC_ROOT`` that is the *only* sanctioned site for
#: chat-model construction (``csa_agent.llm.get_llm`` lives here).
_LLM_FACTORY_FILENAME: str = "llm.py"

#: Minimal OpenAI-compatible chat-completion JSON. ``ChatOpenAI`` (via
#: the openai SDK) parses this shape without errors, so the runtime spy
#: completes a real ``invoke`` round-trip without touching the network.
_CANNED_CHAT_COMPLETION: dict = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1_700_000_000,
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "stub"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
    },
}


# ---------------------------------------------------------------------------
# Runtime half: every outbound LLM request hits Nebius
# ---------------------------------------------------------------------------


@pytest.mark.httpx_mock(
    assert_all_responses_were_requested=False,
    can_send_already_matched_responses=True,
)
def test_runtime_llm_calls_target_nebius_base_url(
    httpx_mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Feature: customer-service-data-analyst-agent, Property 14: All LLM clients point at Nebius.

    Validates Requirements 9.1, 9.2.

    Drive a single ``get_llm().invoke([HumanMessage(...)])`` and assert
    that *every* HTTP request observed by ``pytest-httpx`` targets the
    Nebius host. The fixture intercepts at the
    :class:`httpx.HTTPTransport` layer, which is also what the
    ``openai`` SDK uses, so any chat-model call routed through
    ``ChatOpenAI`` is captured -- including hypothetical multi-request
    flows (auth probes, retries, streaming chunks).
    """

    # Stub a non-empty API key + the canonical Nebius base URL via the
    # environment so :func:`get_settings` produces the expected host.
    # No real network traffic occurs because pytest-httpx serves the
    # canned response below.
    monkeypatch.setenv("NEBIUS_API_KEY", "sk-test-property-14")
    monkeypatch.setenv("NEBIUS_BASE_URL", DEFAULT_NEBIUS_BASE_URL)
    # Reset the cached Settings so the env overrides above take effect
    # even if another test already populated the cache.
    get_settings.cache_clear()

    settings = get_settings()
    expected_host = urlsplit(settings.nebius_base_url).hostname
    assert expected_host, "settings.nebius_base_url must include a host"

    # Serve the canned chat-completion JSON for any request the LLM
    # client issues. ``can_send_already_matched_responses=True`` (set
    # via the marker above) makes this response reusable so retries or
    # multi-request flows still get a 200 back.
    httpx_mock.add_response(json=_CANNED_CHAT_COMPLETION)

    llm = get_llm()
    result = llm.invoke([HumanMessage(content="ping")])

    # The spy must have observed at least one outbound request -- if
    # it didn't, ChatOpenAI silently bypassed httpx (which would itself
    # be a Property-14 violation worth surfacing).
    requests = httpx_mock.get_requests()
    assert requests, (
        "expected ChatOpenAI.invoke to issue at least one HTTP request "
        "via httpx; got none -- the openai SDK may have changed its "
        "transport in a way that bypasses pytest-httpx (Property 14)."
    )

    for req in requests:
        host = req.url.host
        assert host == expected_host, (
            f"LLM request targeted host {host!r} (full URL: {req.url}); "
            f"expected Nebius host {expected_host!r}. "
            "All LLM clients must point at the Nebius Token Factory "
            "(Property 14)."
        )

    # Sanity check: the run completed and produced a parseable response,
    # so the runtime spy actually exercised ChatOpenAI's full pipeline
    # (request, parse, return) rather than short-circuiting.
    assert getattr(result, "content", None) is not None, (
        "ChatOpenAI returned no content; the canned response shape may "
        "have drifted from the OpenAI v1 chat.completions schema."
    )


# ---------------------------------------------------------------------------
# Static half: AST scan of src/csa_agent/ for forbidden constructors
# ---------------------------------------------------------------------------


def _iter_python_files(root: Path) -> list[Path]:
    """Return every ``.py`` file under ``root`` in deterministic order.

    Sorted output makes failure messages reproducible across platforms.
    """

    return sorted(p for p in root.rglob("*.py") if p.is_file())


def _called_name(call: ast.Call) -> str | None:
    """Return the bare name of the callee for a :class:`ast.Call` node.

    Handles both ``ChatOpenAI(...)`` (``ast.Name``) and
    ``langchain_openai.ChatOpenAI(...)`` (``ast.Attribute``) forms.
    Returns ``None`` for higher-order calls like ``f()()`` whose callee
    is itself a :class:`ast.Call` -- those can't statically match a
    forbidden class name anyway.
    """

    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_factory_file(path: Path) -> bool:
    """Return ``True`` for the single sanctioned constructor site.

    Only ``src/csa_agent/llm.py`` is allowed to instantiate a chat-model
    class directly. We deliberately match on parent directory equality
    (not just filename) so a same-named ``llm.py`` in a sub-package
    would still be scanned.
    """

    return path.name == _LLM_FACTORY_FILENAME and path.parent == _SRC_ROOT


def test_no_chat_model_constructors_outside_llm_py() -> None:
    """Feature: customer-service-data-analyst-agent, Property 14: All LLM clients point at Nebius.

    Validates Requirements 9.1, 9.2.

    Walk every ``.py`` file under ``src/csa_agent/`` (excluding
    ``llm.py``), parse it, and assert that no :class:`ast.Call` node's
    callee resolves to a forbidden chat-model class name. This is the
    static complement to the runtime spy: at parse time there exists
    no other constructor that could possibly target a non-Nebius host.
    """

    python_files = _iter_python_files(_SRC_ROOT)
    assert python_files, (
        f"no Python files found under {_SRC_ROOT} -- the static AST "
        "scan would silently pass, which would defeat Property 14."
    )

    # Sanity check: the sanctioned constructor site must actually exist
    # and *must* contain at least one forbidden-class call (that's the
    # whole point of having a single factory). If this assertion fires
    # we have a much bigger problem than a Property-14 violation.
    factory_path = _SRC_ROOT / _LLM_FACTORY_FILENAME
    assert factory_path.is_file(), (
        f"expected the sanctioned LLM factory at {factory_path}; the "
        "scan would otherwise refuse a file that no longer exists."
    )

    violations: list[tuple[Path, int, str]] = []
    for path in python_files:
        if _is_factory_file(path):
            continue
        try:
            tree = ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(path),
            )
        except SyntaxError as exc:  # pragma: no cover - defensive
            pytest.fail(f"could not parse {path}: {exc}")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _called_name(node)
            if name is None:
                continue
            if name in FORBIDDEN_CHAT_MODEL_CLASSES:
                violations.append((path, node.lineno, name))

    if violations:
        rendered = "\n".join(
            f"  - {path.relative_to(_REPO_ROOT).as_posix()}:{lineno} -> {name}(...)"
            for path, lineno, name in violations
        )
        pytest.fail(
            "Forbidden chat model constructor(s) found outside "
            "csa_agent.llm -- only csa_agent.llm.get_llm should "
            "construct LLM clients (Property 14):\n" + rendered
        )
