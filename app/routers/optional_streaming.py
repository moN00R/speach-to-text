"""
Optional Phase – Streaming
WebSocket /ws/stream
    • Create streaming API
    • Send audio chunks
    • Handle real-time communication
"""

import json
import uuid
import base64
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import settings
from app.utils import remove_temp_file
from app.services import whisper_stt, ollama_llm, piper_tts

router = APIRouter(tags=["Optional – Streaming"])


@router.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    """
    WebSocket streaming pipeline.

    Client protocol:
    ─────────────────
    1. Client sends a JSON message:
       {"type": "audio", "data": "<base64-encoded-audio>", "format": "wav"}

    2. Server streams back multiple JSON messages:
       {"type": "stt",       "text": "transcribed user text"}
       {"type": "llm_token", "token": "partial token"}
       {"type": "llm_done",  "full_text": "complete answer"}
       {"type": "tts_ready", "audio": "<base64-wav>"}    ← optional
       {"type": "error",     "message": "what went wrong"}
    """
    await ws.accept()

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            if msg.get("type") != "audio":
                await ws.send_json({"type": "error", "message": "Expected type=audio"})
                continue

            # ── decode & save incoming audio ────────────────────────────
            audio_bytes = base64.b64decode(msg["data"])
            ext = f".{msg.get('format', 'wav')}"
            temp_in = settings.TEMP_AUDIO_DIR / f"ws_in_{uuid.uuid4().hex}{ext}"

            with open(temp_in, "wb") as f:
                f.write(audio_bytes)

            try:
                # ── 1) STT ──────────────────────────────────────────────
                stt_result = await whisper_stt.transcribe(temp_in)
                user_text = stt_result["text"]

                await ws.send_json({"type": "stt", "text": user_text})

                if not user_text.strip():
                    await ws.send_json({"type": "error", "message": "No speech detected."})
                    continue

                # ── 2) LLM – stream tokens ──────────────────────────────
                full_response_parts: list[str] = []
                async for token in ollama_llm.generate_stream(user_text):
                    full_response_parts.append(token)
                    await ws.send_json({"type": "llm_token", "token": token})

                full_response = "".join(full_response_parts)
                await ws.send_json({"type": "llm_done", "full_text": full_response})

                # ── 3) TTS (optional – client can request it) ───────────
                if msg.get("tts", True):
                    temp_out = settings.TEMP_AUDIO_DIR / f"ws_out_{uuid.uuid4().hex}.wav"
                    try:
                        await piper_tts.synthesize(full_response, temp_out)
                        audio_b64 = base64.b64encode(temp_out.read_bytes()).decode()
                        await ws.send_json({"type": "tts_ready", "audio": audio_b64})
                    except Exception as tts_err:
                        await ws.send_json({
                            "type": "error",
                            "message": f"TTS failed: {tts_err}",
                        })
                    finally:
                        remove_temp_file(temp_out)

            finally:
                remove_temp_file(temp_in)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
