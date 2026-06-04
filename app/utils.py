"""Shared helpers – temp audio file management, validation, etc."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import UploadFile, HTTPException

from app.config import settings


def validate_audio(file: UploadFile) -> None:
    """Raise 400 if the upload isn't an accepted audio type."""
    ct = file.content_type or ""
    if ct not in settings.ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio type '{ct}'. Accepted: {settings.ALLOWED_AUDIO_TYPES}",
        )


async def save_temp_audio(file: UploadFile) -> Path:
    """
    Persist an uploaded audio file to disk and return its path.
    The caller is responsible for deleting it afterwards.
    """
    ext = Path(file.filename or "audio.wav").suffix or ".wav"
    filename = f"{uuid.uuid4().hex}{ext}"
    path = settings.TEMP_AUDIO_DIR / filename
    content = await file.read()

    # Size check
    if len(content) > settings.MAX_AUDIO_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file exceeds {settings.MAX_AUDIO_SIZE_MB} MB limit.",
        )

    with open(path, "wb") as f:
        f.write(content)

    return path


def remove_temp_file(path: Path | str) -> None:
    """Silently delete a temporary file."""
    try:
        os.remove(path)
    except OSError:
        pass
