"""
Configuration settings for the Speech Translation System
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
VOICE_SAMPLES_DIR = DATA_DIR / "voice_samples"
SAMPLES_DIR = DATA_DIR / "samples"

# Ensure directories exist
for dir_path in [DATA_DIR, MODELS_DIR, VOICE_SAMPLES_DIR, SAMPLES_DIR]:
    dir_path.mkdir(exist_ok=True)

# Speech Recognition Settings
WHISPER_MODEL_SIZE = "small"  # Options: tiny, base, small, medium, large (small recommended for Hindi)
WHISPER_DEVICE = "auto"  # auto, cpu, cuda

# Translation Settings
DEFAULT_TRANSLATION_SERVICE = "google"  # google, local
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish", 
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "hi": "Hindi"
}

# Voice Cloning Settings
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
VOICE_CLONE_SAMPLES_MIN = 3  # Minimum voice samples needed
VOICE_CLONE_DURATION_MIN = 10  # Minimum duration in seconds

# Audio Processing Settings
SAMPLE_RATE = 22050
MAX_AUDIO_DURATION = 300  # 5 minutes maximum
AUDIO_FORMATS = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]

# API Settings
API_HOST = "localhost"
API_PORT = 8000
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"