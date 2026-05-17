"""Nebius Token Factory LLM factory.

This module provides :func:`get_llm`, the **only** constructor for chat
language-model clients in the codebase. All LLM calls in the system —
query routing, ReAct reasoning, summarization, and the optional query
recommender — go through this factory.

Centralising LLM construction here enforces the LLM-provider invariant
required by the spec:

* Requirement 9.1: every LLM call goes through the Nebius Token Factory.
* Requirement 9.2: no other LLM provider API (OpenAI, Anthropic,
  HuggingFace Inference, etc.) is called directly.

This module is the implementation site of **Property 14** ("All LLM
clients point at Nebius"). The property is verified two ways:

1. Runtime: a network spy (``pytest-httpx``) asserts that every outbound
   LLM request issued during a graph run targets ``settings.nebius_base_url``.
2. Static: an AST scan of ``src/csa_agent/`` rejects any direct
   instantiation of ``ChatOpenAI`` (or other chat-model classes) outside
   this file.

DO NOT instantiate ``ChatOpenAI`` (or any other chat model) elsewhere in
the codebase. If a new use case needs an LLM, call :func:`get_llm`.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from .config import get_settings


def get_llm(
    temperature: float = 0.0,
    model: str | None = None,
) -> ChatOpenAI:
    """Return a :class:`ChatOpenAI` client bound to the Nebius Token Factory.

    The returned client uses Nebius's OpenAI-compatible endpoint, so it
    works unchanged with LangChain's tool-calling, structured-output, and
    streaming APIs.

    Parameters
    ----------
    temperature:
        Sampling temperature. Defaults to ``0.0`` so analytical responses
        are reproducible; callers that need creativity (e.g. the optional
        query recommender) can pass a higher value.
    model:
        Optional override for the model identifier. When ``None``, the
        default from :class:`~csa_agent.config.Settings.nebius_model` is
        used.

    Returns
    -------
    ChatOpenAI
        A configured chat-model client whose ``base_url`` always points
        at the Nebius Token Factory and whose ``api_key`` is sourced from
        :class:`~csa_agent.config.Settings`.

    Notes
    -----
    This is the **only** LLM constructor in the codebase (Property 14).
    """

    settings = get_settings()
    return ChatOpenAI(
        base_url=settings.nebius_base_url,
        api_key=settings.nebius_api_key,
        model=model or settings.nebius_model,
        temperature=temperature,
    )


__all__ = ["get_llm"]
