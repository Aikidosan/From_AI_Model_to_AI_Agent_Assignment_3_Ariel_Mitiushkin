"""FastMCP server exposing the dataset tools as MCP-compatible endpoints.

Run with::

    python mcp_server.py

Configuration (environment variables):

==============  ===================================  ====================
Variable        Purpose                              Default
==============  ===================================  ====================
MCP_TRANSPORT   ``stdio`` or ``sse``                 ``stdio``
MCP_HOST        Bind host (used when transport=sse)  ``0.0.0.0``
MCP_PORT        Bind port (used when transport=sse)  ``8000``
==============  ===================================  ====================

Design notes:

* Five core tools are exposed (Requirement 8.1): ``list_categories``,
  ``count_rows``, ``show_examples``, ``filter_by_category``, and
  ``get_intent_distribution``. The Pydantic input schemas from
  ``csa_agent.tools.schemas`` are mirrored on each MCP tool's signature so
  FastMCP's automatic schema-from-annotations matches the LangChain side
  exactly (Requirement 8.5 -- invalid inputs surface as structured Pydantic
  validation errors).
* Each MCP tool delegates to the *same* underlying callable that the
  LangChain ReAct agent uses, by reaching into the ``BaseTool.func``
  attribute of the tools returned by :func:`csa_agent.tools.core.build_tools`.
  This is what guarantees Property 15 (MCP tool calls equal direct tool
  calls) -- there is exactly one implementation, exposed two ways.
* The dataset is loaded once at module import via :func:`get_dataset` and
  shared across all tool calls (Requirement 1.5).
* The file lives at the repo root rather than under ``src/csa_agent/``, so
  it inserts ``src`` into ``sys.path`` to make ``import csa_agent.*`` work
  whether or not the package is installed.
"""

from __future__ import annotations

import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Make ``src/`` importable when running ``python mcp_server.py`` directly.
# This keeps the project usable without an editable install.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from fastmcp import FastMCP  # noqa: E402

from csa_agent.dataset import get_dataset  # noqa: E402
from csa_agent.tools.core import build_tools  # noqa: E402


# ---------------------------------------------------------------------------
# Build the underlying tool implementations once, at import time.
# ---------------------------------------------------------------------------

_DATAFRAME = get_dataset()
_TOOL_REGISTRY = {tool.name: tool for tool in build_tools(_DATAFRAME)}

# Resolve the underlying callables that the LangChain ``@tool`` decorator
# wraps. ``BaseTool.func`` is the original function before decoration, so
# calling it bypasses LangChain's invoke-style validation and gives us the
# *exact* same code path that the agent observes.
_list_categories_impl = _TOOL_REGISTRY["list_categories"].func
_count_rows_impl = _TOOL_REGISTRY["count_rows"].func
_show_examples_impl = _TOOL_REGISTRY["show_examples"].func
_filter_by_category_impl = _TOOL_REGISTRY["filter_by_category"].func
_get_intent_distribution_impl = _TOOL_REGISTRY["get_intent_distribution"].func


# ---------------------------------------------------------------------------
# FastMCP server: read transport/host/port up-front so the constructor sees
# the values when SSE is enabled.
# ---------------------------------------------------------------------------

_DEFAULT_TRANSPORT = "stdio"
_DEFAULT_HOST = "0.0.0.0"
_DEFAULT_PORT = 8000


def _read_env(name: str, default: str) -> str:
    """Return ``name`` from env, treating empty/whitespace as missing."""

    raw = os.environ.get(name)
    if raw is None:
        return default
    trimmed = raw.strip()
    return trimmed or default


def _read_port(name: str, default: int) -> int:
    """Return an integer port from env or ``default`` on parse failure."""

    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        sys.stderr.write(
            f"Invalid integer value for {name}: {raw!r}. "
            f"Falling back to default port {default}.\n"
        )
        return default


_transport = _read_env("MCP_TRANSPORT", _DEFAULT_TRANSPORT).lower()
_host = _read_env("MCP_HOST", _DEFAULT_HOST)
_port = _read_port("MCP_PORT", _DEFAULT_PORT)

# Pass host/port to the constructor so SSE transports pick them up; FastMCP
# ignores them when running over stdio.
mcp: FastMCP = FastMCP("csa-agent", host=_host, port=_port)


# ---------------------------------------------------------------------------
# Tool registrations.
#
# Each MCP tool re-declares the parameter signature from the matching
# Pydantic schema in ``csa_agent.tools.schemas``. FastMCP infers the input
# JSON schema from these annotations + defaults, which is sufficient to
# enforce the same validation rules the LangChain side gets via its explicit
# ``args_schema`` -- in particular, ``show_examples.n`` is constrained to
# ``[1, 50]`` via ``pydantic.Field`` annotation.
# ---------------------------------------------------------------------------


@mcp.tool()
def list_categories() -> list[str]:
    """List the distinct category values present in the dataset."""

    return _list_categories_impl()


@mcp.tool()
def count_rows(
    category: str | None = None,
    intent: str | None = None,
) -> int | dict[str, Any]:
    """Count rows matching the optional category and/or intent filters.

    Returns the integer count of matching rows, or a structured error
    dict (``{"error": ..., "message": ..., "value": ...}``) when the
    supplied category or intent is not present in the dataset.
    """

    return _count_rows_impl(category=category, intent=intent)


@mcp.tool()
def show_examples(
    category: str | None = None,
    intent: str | None = None,
    n: int = 5,
) -> list[str] | dict[str, Any]:
    """Return up to ``n`` representative utterances for the given filters.

    ``n`` must be between 1 and 50 inclusive. Returns a structured error
    dict when the supplied category or intent is not found.
    """

    return _show_examples_impl(category=category, intent=intent, n=n)


@mcp.tool()
def filter_by_category(category: str) -> list[dict[str, Any]] | dict[str, Any]:
    """Return rows whose ``category`` column equals ``category``.

    At most 100 rows are returned to keep payload sizes bounded; use
    ``count_rows`` for the full total. Returns a structured error dict
    when ``category`` is not found.
    """

    return _filter_by_category_impl(category=category)


@mcp.tool()
def get_intent_distribution(category: str) -> dict[str, int] | dict[str, Any]:
    """Return ``intent -> count`` for every intent within ``category``.

    Returns a structured error dict when ``category`` is not found.
    """

    return _get_intent_distribution_impl(category=category)


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the FastMCP server with the configured transport.

    ``MCP_TRANSPORT`` selects ``stdio`` (default) or ``sse``; for ``sse``,
    ``MCP_HOST`` and ``MCP_PORT`` control the bind address.
    """

    if _transport == "sse":
        sys.stderr.write(
            f"Starting FastMCP server (transport=sse, host={_host}, port={_port})\n"
        )
        mcp.run(transport="sse")
    else:
        # ``stdio`` is the FastMCP default and the simplest transport for
        # local clients. We pass it explicitly for clarity.
        sys.stderr.write("Starting FastMCP server (transport=stdio)\n")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
