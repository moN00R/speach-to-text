"""
Phase 1 – Speech To Text
POST /voice-recognition
    • Receive audio file
    • Send audio to Whisper
    • Return recognized text
    • Handle temporary files
"""

from fastapi import APIRouter, UploadFile, File, Query
from typing import Optional

from app.utils import validate_audio, save_temp_audio, remove_temp_file
from app.services import whisper_stt


router = APIRouter(tags=["Phase 1 – Speech To Text"])


@router.post("/voice-recognition")
async def voice_recognition(
    audio: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[str] = Query(None, description="Language code, e.g. 'ar', 'en'. Defaults to Arabic."),
):
    """
    Upload an audio file → convert to text via Whisper → return the text.
    """
    validate_audio(audio)
    temp_path = await save_temp_audio(audio)

    try:
        result = await whisper_stt.transcribe(temp_path, language=language)
        return {
            "success": True,
            "text": result["text"],
            "language": result["language"],
            "language_probability": result["language_probability"],
            "segments": result["segments"],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        remove_temp_file(temp_path)
