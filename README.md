# Offline AI Voice Assistant – Backend

FastAPI backend for the SharafAI Offline Voice Assistant project.

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 2. Install backend dependencies
pip install -r requirements.txt

# 3. Copy env and adjust if needed
cp .env.example .env

# 4. Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## AI Dependencies (installed by AI Engineer)

The backend calls these local services. They must be running before you test the endpoints.

| Service | Install command | What it does |
|---------|----------------|--------------|
| **faster-whisper** | `pip install faster-whisper` | Speech-to-Text |
| **Ollama** | [ollama.com](https://ollama.com) then `ollama pull gemma:2b` | LLM |
| **Piper** | [github.com/rhasspy/piper](https://github.com/rhasspy/piper) | Text-to-Speech |

---

## API Endpoints

### Phase 1 – Speech To Text
```
POST /voice-recognition
  Form-data: audio (file)
  Query:     language (optional, default "ar")
  Returns:   { text, language, segments }
```

### Phase 2 – Text Assistant
```
POST /assistant
  Form-data: audio (file)
  Query:     language, system_prompt (both optional)
  Returns:   { user_text, assistant_response }
```

### Phase 3 – Voice Assistant
```
POST /speech-assistant
  Form-data: audio (file)
  Query:     language, system_prompt, return_audio (default true)
  Returns:   WAV audio file (or JSON if return_audio=false)

POST /speech-assistant/json
  Same input → Returns JSON + audio_url

GET /audio/{audio_id}
  Download a generated audio file
```

### Optional – Streaming (WebSocket)
```
WS /ws/stream
  Send:    {"type":"audio", "data":"<base64>", "format":"wav", "tts": true}
  Receive: stt → llm_token (×N) → llm_done → tts_ready
```

---

## Project Structure

```
offline-ai-backend/
├── app/
│   ├── main.py              ← FastAPI app + CORS + lifespan
│   ├── config.py            ← Settings (reads .env)
│   ├── utils.py             ← Audio validation & temp file helpers
│   ├── routers/
│   │   ├── phase1_stt.py        ← POST /voice-recognition
│   │   ├── phase2_assistant.py  ← POST /assistant
│   │   ├── phase3_voice.py      ← POST /speech-assistant + /audio/{id}
│   │   └── optional_streaming.py← WS  /ws/stream
│   └── services/
│       ├── whisper_stt.py       ← Whisper wrapper
│       ├── ollama_llm.py        ← Ollama HTTP client
│       └── piper_tts.py         ← Piper subprocess wrapper
├── temp_audio/              ← Temporary audio files (auto-created)
├── requirements.txt
├── .env.example
└── README.md
```

## Testing with curl

```bash
# Phase 1 – transcribe only
curl -X POST http://localhost:8000/voice-recognition \
  -F "audio=@test.wav"

# Phase 2 – ask a question
curl -X POST http://localhost:8000/assistant \
  -F "audio=@question.wav"

# Phase 3 – full voice (returns WAV)
curl -X POST http://localhost:8000/speech-assistant \
  -F "audio=@question.wav" --output response.wav

# Phase 3 – JSON mode
curl -X POST http://localhost:8000/speech-assistant/json \
  -F "audio=@question.wav"
```
