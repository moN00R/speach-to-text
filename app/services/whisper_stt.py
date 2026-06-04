"""
Whisper / faster-whisper Speech-To-Text service.
The AI engineer will install the model; this module wraps it for the API.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)

# ── lazy-loaded model singleton ──────────────────────────────────────
_model = None


def _get_model():
    """Load the faster-whisper model once and cache it."""
    global _model
    if _model is None:
        try:
            from faster_whisper import WhisperModel

            logger.info(
                "Loading Whisper model '%s' on %s (%s) …",
                settings.WHISPER_MODEL_SIZE,
                settings.WHISPER_DEVICE,
                settings.WHISPER_COMPUTE_TYPE,
            )
            _model = WhisperModel(
                settings.WHISPER_MODEL_SIZE,
                device=settings.WHISPER_DEVICE,
                compute_type=settings.WHISPER_COMPUTE_TYPE,
            )
            logger.info("Whisper model loaded successfully.")
        except ImportError:
            raise RuntimeError(
                "faster-whisper is not installed. "
                "Ask the AI engineer to run: pip install faster-whisper"
            )
    return _model


async def transcribe(audio_path: str | Path, language: Optional[str] = None) -> dict:
    """
    Transcribe an audio file to text.

    Returns
    -------
    dict  {"text": str, "language": str, "segments": list[dict]}
    """
    import asyncio

    lang = language or settings.WHISPER_LANGUAGE
    model = _get_model()

    # Run the blocking transcription in a thread so we don't block the event loop
    def _run():
        segments_gen, info = model.transcribe(
            str(audio_path),
            language=lang,
            beam_size=5,
            vad_filter=True,           # skip silence
        )
        segments = []
        full_text_parts = []
        for seg in segments_gen:
            segments.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            })
            full_text_parts.append(seg.text.strip())

        return {
            "text": " ".join(full_text_parts),
            "language": info.language,
            "language_probability": round(info.language_probability, 2),
            "segments": segments,
        }

    return await asyncio.get_event_loop().run_in_executor(None, _run)
