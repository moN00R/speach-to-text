"""
Offline AI Voice Assistant – Backend (FastAPI)
==============================================
Run:  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Phases
------
1. POST /voice-recognition      → Speech-to-Text (Whisper)
2. POST /assistant               → Text Assistant  (Whisper + Ollama)
3. POST /speech-assistant        → Voice Assistant  (Whisper + Ollama + Piper)
   POST /speech-assistant/json   → same, returns JSON + audio URL
   GET  /audio/{id}              → download generated audio
4. WS   /ws/stream               → Streaming (WebSocket)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import phase1_stt, phase2_assistant, phase3_voice, optional_streaming
from app.services import ollama_llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown events."""
    logging.info("🚀  Offline AI Voice Assistant backend starting …")
    yield
    # Cleanup
    await ollama_llm.close()
    logging.info("👋  Backend shut down cleanly.")


app = FastAPI(
    title="Offline AI Voice Assistant",
    description="Fully offline voice assistant – Whisper · Ollama · Piper",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (allow the frontend to call us) ─────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ─────────────────────────────────────────────────
app.include_router(phase1_stt.router)
app.include_router(phase2_assistant.router)
app.include_router(phase3_voice.router)
app.include_router(optional_streaming.router)


# ── Health check ─────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok", "message": "Offline AI Voice Assistant backend is running."}


@app.get("/health")
async def health():
    """Quick liveness check – does not test AI models."""
    return {"status": "healthy"}
