"""Pydantic input schemas and shared error envelope for the dataset tools.

These models define the typed parameter surface for the seven dataset tools
exposed by the Customer Service Data Analyst Agent. They are consumed by:

- LangChain's ``@tool`` decorator (via ``tools/core.py``) so the agent's
  function-calling loop sees a strict JSON schema for every tool.
- FastMCP (via ``mcp_server.py``) so MCP clients receive the same schema
  and identical validation behavior.
- The router and graph layers, which never instantiate raw dicts.

By validating ``ShowExamplesInput.n`` with ``ge=1, le=50`` here, out-of-range
values are rejected at parse time before any tool function runs (Requirement
3.5). ``ToolError`` is the structured envelope every tool returns when a
category or intent is missing, so the agent observes a recoverable result
instead of an exception (Requirement 3.8).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ListCategoriesInput(BaseModel):
    """No parameters required."""

    pass


class FilterByIntentInput(BaseModel):
    intent: str = Field(..., description="Exact intent name (e.g., 'track_refund')")


class FilterByCategoryInput(BaseModel):
    category: str = Field(..., description="Exact category name (e.g., 'REFUND')")


class CountRowsInput(BaseModel):
    category: str | None = Field(None, description="Optional category filter")
    intent: str | None = Field(None, description="Optional intent filter")


class ShowExamplesInput(BaseModel):
    category: str | None = Field(None)
    intent: str | None = Field(None)
    n: int = Field(5, ge=1, le=50, description="Number of examples (1–50)")


class GetIntentDistributionInput(BaseModel):
    category: str = Field(..., description="Exact category name")


class SummarizeCategoryInput(BaseModel):
    category: str = Field(..., description="Exact category name")


class ToolError(BaseModel):
    """Structured error envelope returned by tools instead of raising.

    Tools return this shape (as a dict) when a category or intent argument
    does not exist in the dataset, allowing the ReAct agent to recover
    (Requirement 3.8).
    """

    error: str
    message: str
    value: str | None = None
