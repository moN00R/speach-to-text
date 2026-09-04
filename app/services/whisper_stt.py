"""Whisper STT wrapper (faster-whisper) – CTranslate2 based, no PyTorch needed."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    """Lazily load the model on first use, so the app can boot without it installed."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        logger.info(
            "Loading faster-whisper model '%s' (%s/%s) ...",
            settings.WHISPER_MODEL_SIZE,
            settings.WHISPER_DEVICE,
            settings.WHISPER_COMPUTE_TYPE,
        )
        _model = WhisperModel(
            settings.WHISPER_MODEL_SIZE,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
        )
    return _model


def _transcribe_sync(path: str, language: Optional[str]) -> dict:
    model = _get_model()
    segments, info = model.transcribe(path, language=language, beam_size=5)

    segment_list = [
        {"start": seg.start, "end": seg.end, "text": seg.text.strip()} for seg in segments
    ]
    text = " ".join(seg["text"] for seg in segment_list).strip()

    return {
        "text": text,
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": segment_list,
    }


async def transcribe(path: Path | str, language: Optional[str] = None) -> dict:
    """Transcribe an audio file to text. Runs the blocking model call in a worker thread."""
    lang = language or settings.WHISPER_LANGUAGE
    return await asyncio.to_thread(_transcribe_sync, str(path), lang)
