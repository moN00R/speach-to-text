"""Ollama LLM client – thin async wrapper around the local Ollama HTTP API."""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(base_url=settings.OLLAMA_BASE_URL, timeout=120.0)
    return _client


def _build_messages(user_text: str, system_prompt: Optional[str]) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt or settings.OLLAMA_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


_CONNECTION_ERROR_MSG = (
    "Could not reach Ollama at {url}. Is `ollama serve` running?"
).format(url=settings.OLLAMA_BASE_URL)


async def generate(user_text: str, system_prompt: Optional[str] = None) -> str:
    """Send a prompt to Ollama and return the full response text."""
    client = _get_client()
    try:
        resp = await client.post(
            "/api/chat",
            json={
                "model": settings.OLLAMA_MODEL,
                "messages": _build_messages(user_text, system_prompt),
                "stream": False,
            },
        )
        resp.raise_for_status()
    except httpx.ConnectError as e:
        raise RuntimeError(_CONNECTION_ERROR_MSG) from e

    data = resp.json()
    return data.get("message", {}).get("content", "").strip()


async def generate_stream(
    user_text: str, system_prompt: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """Stream the response from Ollama token-by-token."""
    client = _get_client()
    try:
        async with client.stream(
            "POST",
            "/api/chat",
            json={
                "model": settings.OLLAMA_MODEL,
                "messages": _build_messages(user_text, system_prompt),
                "stream": True,
            },
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break
    except httpx.ConnectError as e:
        raise RuntimeError(_CONNECTION_ERROR_MSG) from e


async def close() -> None:
    """Close the shared HTTP client (called on app shutdown)."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
