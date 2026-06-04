"""
Phase 2 – Text Assistant
POST /assistant
    • Convert voice to text using Whisper
    • Send text to Ollama
    • Return assistant response
"""

from fastapi import APIRouter, UploadFile, File, Query
from typing import Optional

from app.utils import validate_audio, save_temp_audio, remove_temp_file

router = APIRouter(tags=["Phase 2 – Text Assistant"])


@router.post("/assistant")
async def assistant(
    audio: UploadFile = File(..., description="Audio file with the user's question"),
    language: Optional[str] = Query(None, description="Language code for STT"),
    system_prompt: Optional[str] = Query(None, description="Override the default system prompt"),
):
    """
    User speaks a question via audio →
    Whisper transcribes it →
    Ollama generates a text answer →
    Return both question and answer.
    """
    validate_audio(audio)
    temp_path = await save_temp_audio(audio)

    try:
        # Step 1 – Speech to text
        # stt_result = await whisper_stt.transcribe(temp_path, language=language)
        user_text = "كتير شوب اليوم، نصيحة خليك بالبيت واشرب مي كتير!"  # Mock user text for testing

        if not user_text.strip():
            return {
                "success": False,
                "error": "Could not recognise any speech in the audio.",
            }

        # Step 2 – LLM response
        # answer = await ollama_llm.generate(user_text, system_prompt=system_prompt)
        answer = 'كتير شوب اليوم، نصيحة خليك بالبيت واشرب مي كتير!'  # Mock response for testing

        return {
            "success": True,
            "user_text": user_text,
            "assistant_response": answer,
            "language": "ar",  # Mock language for testing
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        remove_temp_file(temp_path)
