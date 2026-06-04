"""
Piper Text-To-Speech service.
Piper runs as a local subprocess – no network needed.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


def _find_piper() -> str:
    """Locate the piper binary."""
    # Check explicit config first, then PATH
    if Path(settings.PIPER_BINARY).is_file():
        return settings.PIPER_BINARY
    found = shutil.which("piper")
    if found:
        return found
    raise FileNotFoundError(
        "Piper TTS binary not found. "
        "Ask the AI engineer to install it: https://github.com/rhasspy/piper"
    )


async def synthesize(text: str, output_path: str | Path) -> Path:
    """
    Convert text to a WAV audio file using Piper.

    Parameters
    ----------
    text : str          The text to speak.
    output_path : Path  Where to save the .wav file.

    Returns
    -------
    Path to the generated WAV file.
    """
    output_path = Path(output_path)
    piper_bin = _find_piper()

    cmd = [
        piper_bin,
        "--model", settings.PIPER_MODEL,
        "--output_file", str(output_path),
    ]

    logger.info("Running Piper TTS → %s", output_path.name)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await proc.communicate(input=text.encode("utf-8"))

    if proc.returncode != 0:
        err_msg = stderr.decode(errors="replace")
        raise RuntimeError(f"Piper TTS failed (exit {proc.returncode}): {err_msg}")

    if not output_path.exists():
        raise RuntimeError("Piper finished but output file was not created.")

    logger.info("Piper TTS done: %s", output_path.name)
    return output_path
