"""
Phase 3 – Full Voice Assistant
POST /speech-assistant
    • Integrate Whisper + Ollama + Piper
    • Return generated audio response
    • Manage audio files
"""

import uuid

from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from typing import Optional

from app.config import settings
from app.utils import validate_audio, save_temp_audio, remove_temp_file
from app.services import whisper_stt, ollama_llm, piper_tts

router = APIRouter(tags=["Phase 3 – Voice Assistant"])


@router.post("/speech-assistant")
async def speech_assistant(
    audio: UploadFile = File(..., description="Audio file with the user's question"),
    language: Optional[str] = Query(None, description="Language code for STT"),
    system_prompt: Optional[str] = Query(None, description="Override the default system prompt"),
    return_audio: bool = Query(True, description="If False, return JSON only (no audio file)"),
):
    """
    Full voice conversation:
    User audio → Whisper STT → Ollama LLM → Piper TTS → Audio response
    """
    validate_audio(audio)
    input_path = await save_temp_audio(audio)
    output_path = settings.TEMP_AUDIO_DIR / f"response_{uuid.uuid4().hex}.wav"

    try:
        # 1) Speech → Text
        stt_result = await whisper_stt.transcribe(input_path, language=language)
        user_text = stt_result["text"]

        if not user_text.strip():
            return {"success": False, "error": "No speech detected in the audio."}

        # 2) Text → LLM answer
        answer = await ollama_llm.generate(user_text, system_prompt=system_prompt)

        # 3) Answer → Speech
        if return_audio:
            await piper_tts.synthesize(answer, output_path)
            return FileResponse(
                path=str(output_path),
                media_type="audio/wav",
                filename="response.wav",
                headers={
                    "X-User-Text": user_text,
                    "X-Assistant-Text": answer[:500],  # truncate for header safety
                },
                background=BackgroundTask(remove_temp_file, output_path),
            )

        # JSON-only mode (useful for debugging or when frontend handles TTS)
        return {
            "success": True,
            "user_text": user_text,
            "assistant_response": answer,
            "language": stt_result["language"],
        }

    except Exception as e:
        remove_temp_file(output_path)
        return {"success": False, "error": str(e)}
    finally:
        remove_temp_file(input_path)


@router.post("/speech-assistant/json")
async def speech_assistant_json(
    audio: UploadFile = File(..., description="Audio file with the user's question"),
    language: Optional[str] = Query(None, description="Language code for STT"),
    system_prompt: Optional[str] = Query(None, description="Override the default system prompt"),
):
    """
    Same pipeline but always returns JSON with a download URL for the audio.
    Useful if the frontend wants the text AND the audio file path.
    """
    validate_audio(audio)
    input_path = await save_temp_audio(audio)
    audio_id = uuid.uuid4().hex
    output_path = settings.TEMP_AUDIO_DIR / f"response_{audio_id}.wav"

    try:
        stt_result = await whisper_stt.transcribe(input_path, language=language)
        user_text = stt_result["text"]

        if not user_text.strip():
            return {"success": False, "error": "No speech detected."}

        answer = await ollama_llm.generate(user_text, system_prompt=system_prompt)
        await piper_tts.synthesize(answer, output_path)

        return {
            "success": True,
            "user_text": user_text,
            "assistant_response": answer,
            "language": stt_result["language"],
            "audio_url": f"/audio/{audio_id}",
        }
    except Exception as e:
        remove_temp_file(output_path)
        return {"success": False, "error": str(e)}
    finally:
        remove_temp_file(input_path)


@router.get("/audio/{audio_id}")
async def get_audio(audio_id: str):
    """Serve a generated audio response file."""
    path = settings.TEMP_AUDIO_DIR / f"response_{audio_id}.wav"
    if not path.exists():
        return {"success": False, "error": "Audio file not found or expired."}
    return FileResponse(path, media_type="audio/wav", filename="response.wav")
