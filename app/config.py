import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Paths ---
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    TEMP_AUDIO_DIR: Path = BASE_DIR / "temp_audio"

    # --- Whisper (faster-whisper) ---
    WHISPER_MODEL_SIZE: str = "tiny"        # tiny, base, small, medium, large-v3
    WHISPER_DEVICE: str = "cpu"             # cpu or cuda
    WHISPER_COMPUTE_TYPE: str = "int8"      # int8 for CPU, float16 for GPU
    WHISPER_LANGUAGE: str = "ar"            # Arabic by default

    # --- Ollama ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma:2b"          # or qwen:1.8b, etc.
    OLLAMA_SYSTEM_PROMPT: str = (
        "You are a helpful Arabic-speaking voice assistant. "
        "Answer concisely and clearly. You can also respond in English if the user speaks English."
    )

    # --- Piper TTS ---
    PIPER_BINARY: str = "piper"             # path to piper executable
    PIPER_MODEL: str = "ar_JO-kareem-medium.onnx"  # Arabic voice model
    PIPER_SAMPLE_RATE: int = 22050

    # --- Server ---
    ALLOWED_AUDIO_TYPES: list[str] = [
        "audio/wav", "audio/wave", "audio/x-wav",
        "audio/mpeg", "audio/mp3", "audio/mp4",
        "audio/ogg", "audio/webm", "audio/flac",
        "application/octet-stream",                 # fallback for some browsers
    ]
    MAX_AUDIO_SIZE_MB: int = 25

    model_config = {"env_prefix": "APP_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
os.makedirs(settings.TEMP_AUDIO_DIR, exist_ok=True)
