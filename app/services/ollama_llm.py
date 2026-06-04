"""
Ollama LLM service – talks to the locally-running Ollama server.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=settings.OLLAMA_BASE_URL,
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
        )
    return _client


async def generate(prompt: str, system_prompt: str | None = None) -> str:
    """
    Send a prompt to Ollama and return the full response text.
    """
    client = _get_client()
    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": prompt,
        "system": system_prompt or settings.OLLAMA_SYSTEM_PROMPT,
        "stream": False,
    }

    try:
        resp = await client.post("/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except httpx.ConnectError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {settings.OLLAMA_BASE_URL}. "
            "Make sure Ollama is running: ollama serve"
        )
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Ollama returned {exc.response.status_code}: {exc.response.text}")


async def generate_stream(prompt: str, system_prompt: str | None = None) -> AsyncGenerator[str, None]:
    """
    Stream tokens from Ollama one chunk at a time.
    """
    client = _get_client()
    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": prompt,
        "system": system_prompt or settings.OLLAMA_SYSTEM_PROMPT,
        "stream": True,
    }

    try:
        async with client.stream("POST", "/api/generate", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                import json
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break
    except httpx.ConnectError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {settings.OLLAMA_BASE_URL}. "
            "Make sure Ollama is running: ollama serve"
        )


async def close():
    """Shut down the HTTP client gracefully."""
    global _client
    if _client:
        await _client.aclose()
        _client = None
