"""
Test Configuration and Utilities

This module provides configuration and utilities for testing the
speech translation system.
"""

import pytest
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
import logging

# Test configuration
TEST_CONFIG = {
    'sample_rate': 22050,
    'test_duration': 3.0,  # seconds
    'test_text': "Hello, this is a test message for speech translation.",
    'source_language': 'en',
    'target_language': 'es',
    'expected_translation_keywords': ['hola', 'mensaje', 'prueba'],  # Spanish keywords
}

# Disable verbose logging during tests
logging.getLogger('TTS').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)


def create_test_audio(
    duration: float = TEST_CONFIG['test_duration'],
    sample_rate: int = TEST_CONFIG['sample_rate'],
    frequency: float = 440.0
) -> np.ndarray:
    """
    Create a test audio signal (sine wave).
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        frequency: Frequency of sine wave in Hz
        
    Returns:
        Audio data as numpy array
    """
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)  # Amplitude 0.3 to avoid clipping
    return audio.astype(np.float32)


def create_test_audio_file(
    filepath: Path,
    duration: float = TEST_CONFIG['test_duration'],
    sample_rate: int = TEST_CONFIG['sample_rate']
) -> Path:
    """
    Create a test audio file.
    
    Args:
        filepath: Path for the output file
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Path to created file
    """
    audio = create_test_audio(duration, sample_rate)
    sf.write(str(filepath), audio, sample_rate)
    return filepath


class TestAudioManager:
    """Manages test audio files for testing."""
    
    def __init__(self):
        self.temp_dir = None
        self.created_files = []
    
    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup is handled automatically by tempfile
        pass
    
    def create_audio_file(
        self, 
        filename: str = "test_audio.wav",
        duration: float = TEST_CONFIG['test_duration']
    ) -> Path:
        """Create a test audio file in temporary directory."""
        filepath = Path(self.temp_dir) / filename
        create_test_audio_file(filepath, duration)
        self.created_files.append(filepath)
        return filepath
    
    def create_voice_sample(
        self, 
        filename: str = "voice_sample.wav",
        duration: float = 10.0  # Voice samples should be longer
    ) -> Path:
        """Create a voice sample file for testing."""
        return self.create_audio_file(filename, duration)
    
    def get_temp_dir(self) -> Path:
        """Get temporary directory path."""
        return Path(self.temp_dir)


@pytest.fixture
def audio_manager():
    """Pytest fixture for test audio management."""
    with TestAudioManager() as manager:
        yield manager


@pytest.fixture
def test_audio_file(audio_manager):
    """Pytest fixture for a single test audio file."""
    return audio_manager.create_audio_file()


@pytest.fixture
def voice_sample_file(audio_manager):
    """Pytest fixture for a voice sample file."""
    return audio_manager.create_voice_sample()


@pytest.fixture
def batch_audio_files(audio_manager):
    """Pytest fixture for multiple test audio files."""
    files = []
    for i in range(3):
        files.append(audio_manager.create_audio_file(f"test_audio_{i}.wav"))
    return files


def skip_if_no_gpu():
    """Skip test if no GPU is available."""
    import torch
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="GPU not available"
    )


def skip_if_no_internet():
    """Skip test if no internet connection."""
    import socket
    
    def has_internet():
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False
    
    return pytest.mark.skipif(
        not has_internet(),
        reason="Internet connection not available"
    )


# Mock classes for testing without loading actual models
class MockSpeechRecognizer:
    """Mock speech recognizer for testing."""
    
    def __init__(self, *args, **kwargs):
        self.model_loaded = True
    
    def load_model(self):
        pass
    
    def transcribe(self, audio_path, **kwargs):
        return {
            'text': TEST_CONFIG['test_text'],
            'language': TEST_CONFIG['source_language'],
            'segments': [],
            'confidence': 0.95,
            'audio_path': str(audio_path)
        }
    
    def detect_language(self, audio_path):
        return {
            'detected_language': TEST_CONFIG['source_language'],
            'confidence': 0.95,
            'top_languages': [
                {'language': TEST_CONFIG['source_language'], 'confidence': 0.95}
            ]
        }
    
    def get_model_info(self):
        return {
            'model_size': 'base',
            'device': 'cpu',
            'model_loaded': True
        }


class MockTranslationService:
    """Mock translation service for testing."""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def translate(self, text, source_lang, target_lang, **kwargs):
        # Simple mock translation
        translated = f"Translated: {text} ({source_lang} -> {target_lang})"
        return {
            'text': text,
            'translated_text': translated,
            'source_language': source_lang,
            'target_language': target_lang,
            'confidence': 0.90,
            'engine': 'mock'
        }
    
    def detect_language(self, text):
        return {
            'language': TEST_CONFIG['source_language'],
            'confidence': 0.90,
            'engine': 'mock'
        }
    
    def get_supported_languages(self):
        return {'en': 'English', 'es': 'Spanish', 'fr': 'French'}


class MockVoiceCloner:
    """Mock voice cloner for testing."""
    
    def __init__(self, *args, **kwargs):
        self.voice_samples = {}
    
    def load_model(self):
        pass
    
    def register_voice(self, speaker_name, voice_samples, **kwargs):
        self.voice_samples[speaker_name] = {
            'samples': voice_samples,
            'num_samples': len(voice_samples)
        }
        return {
            'speaker_name': speaker_name,
            'num_samples': len(voice_samples),
            'total_duration': 30.0,
            'status': 'registered'
        }
    
    def clone_voice(self, text, speaker_name, language, output_path=None, **kwargs):
        # Create mock audio output
        if output_path:
            audio = create_test_audio(duration=5.0)
            sf.write(str(output_path), audio, TEST_CONFIG['sample_rate'])
        
        return {
            'text': text,
            'speaker_name': speaker_name,
            'language': language,
            'audio_data': create_test_audio(duration=5.0),
            'sample_rate': TEST_CONFIG['sample_rate'],
            'duration': 5.0,
            'output_path': str(output_path) if output_path else None
        }
    
    def get_registered_speakers(self):
        return list(self.voice_samples.keys())
    
    def get_model_info(self):
        return {
            'model_name': 'mock_tts',
            'device': 'cpu',
            'model_loaded': True,
            'num_registered_speakers': len(self.voice_samples)
        }