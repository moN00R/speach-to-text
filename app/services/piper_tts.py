"""Piper TTS wrapper – shells out to the piper binary."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


async def synthesize(text: str, output_path: Path | str) -> Path:
    """Synthesize `text` to a WAV file at `output_path` using the piper binary."""
    output_path = Path(output_path)
    cmd = [
        settings.PIPER_BINARY,
        "--model",
        settings.PIPER_MODEL,
        "--output_file",
        str(output_path),
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Piper binary '{settings.PIPER_BINARY}' not found. "
            "Install it from github.com/rhasspy/piper and/or set APP_PIPER_BINARY."
        ) from e

    _, stderr = await proc.communicate(input=text.encode("utf-8"))

    if proc.returncode != 0:
        raise RuntimeError(f"Piper TTS failed: {stderr.decode(errors='ignore')}")

    if not output_path.exists():
        raise RuntimeError("Piper TTS did not produce an output file.")

    return output_path
