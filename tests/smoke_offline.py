"""Offline smoke tests for the Customer Service Data Analyst Agent.

These tests verify everything that does NOT require a real Nebius API
call: dataset loader, every direct dataset tool, profile round-trip,
checkpointer parent-directory validation, router fallback semantics,
the FastMCP tool functions, and a full LangGraph compile. The router's
LLM call is exercised through a fake chat model so we can assert the
classify_query coercion path without hitting Nebius.

Run with:
    python tests\\smoke_offline.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Make src/ importable and provide a stub API key so config.get_settings
# does not exit on import.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

os.environ.setdefault("NEBIUS_API_KEY", "stub-key-for-offline-tests")

from csa_agent.checkpointer import get_checkpointer  # noqa: E402
from csa_agent.config import get_settings  # noqa: E402
from csa_agent.dataset import REQUIRED_COLUMNS, get_dataset, load_dataset  # noqa: E402
from csa_agent.profile import (  # noqa: E402
    UserProfile,
    load_profile,
    record_topic,
    save_profile,
)
from csa_agent.router import RouteLabel, classify_query  # noqa: E402
from csa_agent.tools.core import build_tools  # noqa: E402
from csa_agent.tools.schemas import ShowExamplesInput, ToolError  # noqa: E402

# Tally of pass/fail so a single failure doesn't abort the whole sweep.
_PASS = 0
_FAIL = 0


def _check(name: str, condition: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  PASS  {name}")
    else:
        _FAIL += 1
        print(f"  FAIL  {name}  {detail}")


# ---------------------------------------------------------------------------
# 1. Settings + dataset
# ---------------------------------------------------------------------------
print("\n[1] Settings + dataset")
settings = get_settings()
_check(
    "Settings loaded with defaults",
    settings.nebius_base_url.startswith("https://api.studio.nebius"),
    f"got {settings.nebius_base_url!r}",
)
_check("Settings.max_iterations == 15", settings.max_iterations == 15)

df = get_dataset()
_check(
    "Dataset loaded with required columns",
    REQUIRED_COLUMNS.issubset(set(df.columns)),
    f"got {sorted(df.columns)}",
)
_check("Dataset has > 1000 rows", len(df) > 1000, f"got {len(df)}")
_check(
    "instruction -> utterance alias applied",
    "utterance" in df.columns and "instruction" not in df.columns,
)


# ---------------------------------------------------------------------------
# 2. Dataset tools (direct calls, no LLM)
# ---------------------------------------------------------------------------
print("\n[2] Dataset tools")
tools = {t.name: t for t in build_tools(df)}
_check(
    "Seven tools registered",
    set(tools.keys())
    == {
        "list_categories",
        "filter_by_intent",
        "filter_by_category",
        "count_rows",
        "show_examples",
        "get_intent_distribution",
        "summarize_category",
    },
    f"got {sorted(tools.keys())}",
)

# Use .invoke() for LangChain BaseTool — that's the supported call surface.
categories = tools["list_categories"].invoke({})
_check(
    "list_categories returns sorted unique strings",
    isinstance(categories, list)
    and len(categories) == len(set(categories))
    and categories == sorted(categories),
)
_check(
    "list_categories matches DataFrame distinct categories",
    set(categories) == {str(v) for v in df["category"].dropna().unique()},
)

# Pick a real category we can reason about.
sample_category = categories[0]
filtered = tools["filter_by_category"].invoke({"category": sample_category})
_check(
    "filter_by_category returns list of dicts",
    isinstance(filtered, list) and all(isinstance(r, dict) for r in filtered),
)
_check("filter_by_category capped at 100", len(filtered) <= 100)

count_full = tools["count_rows"].invoke({})
_check(
    "count_rows() with no filter == len(df)",
    count_full == len(df),
    f"got {count_full} vs {len(df)}",
)

count_cat = tools["count_rows"].invoke({"category": sample_category})
expected_cat = int((df["category"] == sample_category).sum())
_check(
    "count_rows(category=...) equals direct DataFrame filter",
    count_cat == expected_cat,
    f"got {count_cat} vs {expected_cat}",
)

# get_intent_distribution sums to count_rows(category=...).
dist = tools["get_intent_distribution"].invoke({"category": sample_category})
_check(
    "get_intent_distribution sum equals count_rows",
    isinstance(dist, dict) and sum(dist.values()) == expected_cat,
    f"sum={sum(dist.values())} vs {expected_cat}",
)

examples = tools["show_examples"].invoke({"category": sample_category, "n": 3})
_check(
    "show_examples returns list of strings, len <= n",
    isinstance(examples, list)
    and all(isinstance(e, str) for e in examples)
    and len(examples) <= 3,
)

# Unknown category -> structured ToolError dict, never raises.
err = tools["filter_by_category"].invoke({"category": "__BOGUS__"})
_check(
    "Unknown category yields ToolError-shaped dict",
    isinstance(err, dict)
    and err.get("error") == "category_not_found"
    and err.get("value") == "__BOGUS__",
    f"got {err}",
)

# Pydantic schema rejects out-of-range n at validation time.
try:
    ShowExamplesInput(n=99)
    range_rejected = False
except Exception:
    range_rejected = True
_check("ShowExamplesInput rejects n=99", range_rejected)

# ToolError model round-trips.
te = ToolError(error="x", message="y", value="z")
_check("ToolError.model_dump round-trip", te.model_dump()["value"] == "z")


# ---------------------------------------------------------------------------
# 3. Profile store
# ---------------------------------------------------------------------------
print("\n[3] Profile store")
with tempfile.TemporaryDirectory() as tmp:
    profile = load_profile("alice", profile_dir=tmp)
    _check(
        "load_profile returns fresh UserProfile when file absent",
        isinstance(profile, UserProfile) and profile.user_id == "alice",
    )
    profile.name = "Alice"
    profile.preferences["color"] = "blue"
    saved = save_profile(profile, profile_dir=tmp)
    _check("save_profile returns refreshed copy", saved.updated_at >= profile.updated_at)

    reloaded = load_profile("alice", profile_dir=tmp)
    _check("Profile round-trip preserves name", reloaded.name == "Alice")
    _check(
        "Profile round-trip preserves preferences",
        reloaded.preferences == {"color": "blue"},
    )

    # record_topic threshold behaviour.
    p = UserProfile(user_id="bob")
    record_topic(p, "REFUND")
    record_topic(p, "REFUND")
    _check("Topic not frequent yet at count 2", "REFUND" not in p.frequent_topics)
    record_topic(p, "REFUND")
    _check("Topic frequent at count 3", "REFUND" in p.frequent_topics)
    record_topic(p, "REFUND")
    _check(
        "Topic not duplicated past threshold",
        p.frequent_topics.count("REFUND") == 1,
    )
    _check("topic_counts == 4 after 4 calls", p.topic_counts["REFUND"] == 4)

    # On-disk JSON is human-readable.
    p2 = UserProfile(user_id="carol", name="Carol")
    save_profile(p2, profile_dir=tmp)
    raw = json.loads(Path(tmp, "carol.json").read_text(encoding="utf-8"))
    _check("Profile JSON contains user_id", raw.get("user_id") == "carol")


# ---------------------------------------------------------------------------
# 4. Checkpointer
# ---------------------------------------------------------------------------
print("\n[4] Checkpointer")
with tempfile.TemporaryDirectory() as tmp:
    db_path = os.path.join(tmp, "subdir", "ckpt.db")
    with get_checkpointer(db_path) as saver:
        _check("Checkpointer opens with auto-created parent dir", saver is not None)
        _check("Checkpointer DB file created", os.path.isfile(db_path))


# ---------------------------------------------------------------------------
# 5. Router with fake LLM (no Nebius call)
# ---------------------------------------------------------------------------
print("\n[5] Query Router with fake LLM")


class _FakeStructuredLLM:
    """Returns whatever we tell it to return, no network."""

    def __init__(self, payload):
        self._payload = payload

    def invoke(self, _messages):
        return self._payload


class _FakeLLM:
    def __init__(self, payload):
        self._payload = payload

    def with_structured_output(self, _model):
        return _FakeStructuredLLM(self._payload)


from csa_agent.router import _RouterDecision  # noqa: E402  (intentional private import for test)

_check(
    "Structured response -> STRUCTURED",
    classify_query("how many refunds", _FakeLLM(_RouterDecision(label=RouteLabel.STRUCTURED)))
    == RouteLabel.STRUCTURED,
)
_check(
    "Dict fallback path -> UNSTRUCTURED",
    classify_query("summarise", _FakeLLM({"label": "unstructured"})) == RouteLabel.UNSTRUCTURED,
)
_check(
    "Raw string fallback (case insensitive)",
    classify_query("?", _FakeLLM("Out_Of_Scope")) == RouteLabel.OUT_OF_SCOPE,
)


class _BoomLLM:
    def with_structured_output(self, _model):
        raise RuntimeError("simulated network error")


_check(
    "LLM exception -> safe OUT_OF_SCOPE default",
    classify_query("anything", _BoomLLM()) == RouteLabel.OUT_OF_SCOPE,
)


# ---------------------------------------------------------------------------
# 6. MCP server module loads and registers 5 tools
# ---------------------------------------------------------------------------
print("\n[6] FastMCP server module")
import importlib.util  # noqa: E402

mcp_spec = importlib.util.spec_from_file_location("mcp_server_test", _REPO / "mcp_server.py")
mcp_mod = importlib.util.module_from_spec(mcp_spec)
mcp_spec.loader.exec_module(mcp_mod)
_check("mcp_server.py imports cleanly", hasattr(mcp_mod, "mcp"))

# Cross-check MCP-exposed tool surface == direct tool surface for list_categories
mcp_categories = mcp_mod.list_categories.fn() if hasattr(mcp_mod.list_categories, "fn") else mcp_mod._list_categories_impl()
_check(
    "MCP list_categories matches direct tool",
    set(mcp_categories) == set(categories),
)


# ---------------------------------------------------------------------------
# 7. Graph compile (no LLM call yet)
# ---------------------------------------------------------------------------
print("\n[7] Graph compile")
from csa_agent.graph import build_graph  # noqa: E402

graph = build_graph(checkpointer=None)
node_names = set(graph.get_graph().nodes.keys())
expected_nodes = {
    "__start__",
    "__end__",
    "load_user_profile",
    "query_router",
    "react_agent",
    "summarize",
    "decline",
    "update_profile",
}
_check(
    "Compiled graph contains all expected nodes",
    expected_nodes.issubset(node_names),
    f"got {sorted(node_names)}",
)


# ---------------------------------------------------------------------------
# 8. Decline path is pure (no LLM, no tools)
# ---------------------------------------------------------------------------
print("\n[8] Decline path is pure")
from csa_agent.nodes import CANONICAL_REFUSAL, decline_node  # noqa: E402

result = decline_node({})
_check(
    "decline_node emits canonical refusal AIMessage",
    isinstance(result, dict)
    and result.get("messages")
    and result["messages"][0].content == CANONICAL_REFUSAL,
)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n=== Offline smoke results: {_PASS} passed, {_FAIL} failed ===")
sys.exit(0 if _FAIL == 0 else 1)
