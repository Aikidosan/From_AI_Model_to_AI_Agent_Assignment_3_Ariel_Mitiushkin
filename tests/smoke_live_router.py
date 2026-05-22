"""Live router smoke test against Nebius.

Asserts the router classifies three canonical queries correctly:
- "How many refund requests are there?" -> STRUCTURED
- "Summarize the FEEDBACK category"     -> UNSTRUCTURED
- "What is the capital of France?"       -> OUT_OF_SCOPE

This is the smallest live test that proves Nebius is reachable, the
factory's auth/base-url work, and the router prompt produces sane
classifications. No tools are called.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from csa_agent.llm import get_llm
from csa_agent.router import RouteLabel, classify_query

CASES = [
    ("How many refund requests are there?", RouteLabel.STRUCTURED),
    ("Summarize the FEEDBACK category", RouteLabel.UNSTRUCTURED),
    ("What is the capital of France?", RouteLabel.OUT_OF_SCOPE),
]

llm = get_llm()
print(f"Using base_url={llm.openai_api_base!r} model={llm.model_name!r}\n")

failures = 0
for query, expected in CASES:
    actual = classify_query(query, llm)
    ok = actual == expected
    print(f"  {'PASS' if ok else 'FAIL'}  {query!r}\n        expected={expected.value}, got={actual.value}")
    if not ok:
        failures += 1

print(f"\n=== Live router results: {len(CASES) - failures}/{len(CASES)} ===")
sys.exit(0 if failures == 0 else 1)
