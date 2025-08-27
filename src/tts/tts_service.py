"""
Text-to-Speech Service with Multiple Fallback Options

Provides speech synthesis with voice cloning capabilities and fallback voices.
"""

import os
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
import numpy as np
import soundfile as sf


class TextToSpeechService:
    """TTS service with multiple backend options"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.temp_dir = Path(tempfile.gettempdir()) / "speech_translation_tts"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Available TTS engines in order of preference
        self.engines = []
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize available TTS engines"""
        # Try to initialize TTS engines in order of preference
        
        # 1. Try gTTS (Google Text-to-Speech) - requires internet
        try:
            import gtts
            self.engines.append('gtts')
            self.logger.info("✅ gTTS (Google TTS) available")
        except ImportError:
            self.logger.warning("⚠️ gTTS not available")
        
        # 2. Try pyttsx3 (offline TTS)
        try:
            import pyttsx3
            self.engines.append('pyttsx3')
            self.logger.info("✅ pyttsx3 (offline TTS) available")
        except ImportError:
            self.logger.warning("⚠️ pyttsx3 not available")
        
        # 3. Always have mock TTS as final fallback
        self.engines.append('mock')
        self.logger.info("✅ Mock TTS available as fallback")
        
        self.logger.info(f"Available TTS engines: {self.engines}")
    
    def synthesize_speech(
        self, 
        text: str, 
        language: str = "en", 
        voice_sample: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convert text to speech
        
        Args:
            text: Text to synthesize
            language: Target language code
            voice_sample: Path to voice sample for cloning (if supported)
            output_path: Output file path (if None, generates temp file)
            
        Returns:
            Result dictionary with audio file path and metadata
        """
        
        if not output_path:
            output_path = self.temp_dir / f"tts_output_{int(time.time())}.wav"
        
        # Try each TTS engine until one works
        for engine in self.engines:
            try:
                if engine == 'gtts':
                    return self._synthesize_with_gtts(text, language, output_path)
                elif engine == 'pyttsx3':
                    return self._synthesize_with_pyttsx3(text, language, output_path)
                elif engine == 'mock':
                    return self._synthesize_with_mock(text, language, output_path)
            except Exception as e:
                self.logger.warning(f"TTS engine {engine} failed: {str(e)}")
                continue
        
        # If all engines fail
        return {
            'success': False,
            'error': 'All TTS engines failed',
            'audio_path': None,
            'engine': 'none'
        }
    
    def _synthesize_with_gtts(self, text: str, language: str, output_path: str) -> Dict[str, Any]:
        """Use Google Text-to-Speech"""
        try:
            from gtts import gTTS
            import pygame
            import time
            
            # Map common language codes for gTTS
            gtts_lang_map = {
                'hi': 'hi',
                'en': 'en',
                'es': 'es', 
                'fr': 'fr',
                'de': 'de',
                'it': 'it',
                'pt': 'pt',
                'ru': 'ru',
                'ja': 'ja',
                'ko': 'ko',
                'zh': 'zh',
                'ar': 'ar'
            }
            
            gtts_lang = gtts_lang_map.get(language, 'en')
            
            # Create TTS object
            tts = gTTS(text=text, lang=gtts_lang, slow=False)
            
            # Save to temporary MP3 file first
            temp_mp3 = str(output_path).replace('.wav', '.mp3')
            tts.save(temp_mp3)
            
            # Convert MP3 to WAV using pydub
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(temp_mp3)
            audio.export(output_path, format="wav")
            
            # Clean up temp MP3
            os.remove(temp_mp3)
            
            return {
                'success': True,
                'audio_path': str(output_path),
                'engine': 'gTTS (Google)',
                'language': language,
                'duration': len(audio) / 1000.0,  # Duration in seconds
                'sample_rate': audio.frame_rate
            }
            
        except Exception as e:
            raise Exception(f"gTTS synthesis failed: {str(e)}")
    
    def _synthesize_with_pyttsx3(self, text: str, language: str, output_path: str) -> Dict[str, Any]:
        """Use pyttsx3 offline TTS"""
        try:
            import pyttsx3
            
            # Initialize TTS engine
            engine = pyttsx3.init()
            
            # Configure voice properties
            voices = engine.getProperty('voices')
            
            # Try to find appropriate voice for language
            selected_voice = None
            for voice in voices:
                voice_lang = getattr(voice, 'languages', [])
                if language in str(voice_lang).lower() or language == 'en':
                    selected_voice = voice.id
                    break
            
            if selected_voice:
                engine.setProperty('voice', selected_voice)
            
            # Set speech rate and volume
            engine.setProperty('rate', 150)  # Speed of speech
            engine.setProperty('volume', 0.8)  # Volume level (0.0 to 1.0)
            
            # Save to file
            engine.save_to_file(text, str(output_path))
            engine.runAndWait()
            
            # Get audio duration (approximate)
            duration = len(text.split()) * 0.6  # Rough estimate: 0.6 seconds per word
            
            return {
                'success': True,
                'audio_path': str(output_path),
                'engine': 'pyttsx3 (offline)',
                'language': language,
                'duration': duration,
                'sample_rate': 22050  # Default for pyttsx3
            }
            
        except Exception as e:
            raise Exception(f"pyttsx3 synthesis failed: {str(e)}")
    
    def _synthesize_with_mock(self, text: str, language: str, output_path: str) -> Dict[str, Any]:
        """Generate mock audio for demonstration"""
        try:
            import time
            
            # Generate a simple tone sequence based on text
            sample_rate = 22050
            duration = max(2.0, len(text) * 0.1)  # Minimum 2 seconds
            
            t = np.linspace(0, duration, int(duration * sample_rate), False)
            
            # Create a pleasant tone sequence
            # Base frequency varies by language
            base_freq = {
                'hi': 220,  # A3
                'en': 261,  # C4
                'es': 293,  # D4
                'fr': 329,  # E4
                'de': 349,  # F4
            }.get(language, 261)
            
            # Generate harmonics for richer sound
            audio = (
                0.3 * np.sin(2 * np.pi * base_freq * t) +
                0.2 * np.sin(2 * np.pi * base_freq * 1.5 * t) +
                0.1 * np.sin(2 * np.pi * base_freq * 2 * t)
            )
            
            # Add simple envelope (fade in/out)
            fade_samples = int(0.1 * sample_rate)  # 100ms fade
            audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
            audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            
            # Add some variation based on text length
            if len(text) > 50:
                # Longer text gets some frequency modulation
                mod_freq = 2.0  # 2 Hz modulation
                modulation = 1 + 0.1 * np.sin(2 * np.pi * mod_freq * t)
                audio *= modulation
            
            # Normalize
            audio = audio / np.max(np.abs(audio)) * 0.7
            
            # Save as WAV
            sf.write(str(output_path), audio.astype(np.float32), sample_rate)
            
            return {
                'success': True,
                'audio_path': str(output_path),
                'engine': 'Mock TTS (Demo)',
                'language': language,
                'duration': duration,
                'sample_rate': sample_rate,
                'note': 'This is a demo tone. Install gTTS or pyttsx3 for real speech.'
            }
            
        except Exception as e:
            raise Exception(f"Mock TTS failed: {str(e)}")
    
    def clone_voice(
        self, 
        text: str, 
        voice_sample_path: str, 
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Attempt voice cloning (placeholder for future implementation)
        
        Currently falls back to regular TTS with a note about voice cloning.
        """
        
        # For now, use regular TTS but indicate it's attempted cloning
        result = self.synthesize_speech(text, "en", None, output_path)
        
        if result['success']:
            result['note'] = f"Voice cloning attempted using {voice_sample_path}. Currently using fallback TTS."
            result['voice_cloning'] = 'attempted (fallback to TTS)'
        
        return result
    
    def get_available_voices(self) -> Dict[str, Any]:
        """Get information about available voices"""
        voices_info = {
            'engines': self.engines,
            'languages_supported': ['en', 'hi', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'ar'],
            'voice_cloning': 'planned (currently uses fallback)',
            'recommendations': {
                'best_quality': 'gTTS (requires internet)',
                'offline': 'pyttsx3',
                'demo': 'mock (always available)'
            }
        }
        
        # Try to get system voices if pyttsx3 is available
        if 'pyttsx3' in self.engines:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                system_voices = engine.getProperty('voices')
                voices_info['system_voices'] = [
                    {
                        'id': voice.id,
                        'name': voice.name,
                        'languages': getattr(voice, 'languages', [])
                    }
                    for voice in system_voices[:5]  # Limit to first 5
                ]
                engine.stop()
            except:
                pass
        
        return voices_info


def create_tts_service() -> TextToSpeechService:
    """Factory function to create TTS service"""
    return TextToSpeechService()


def test_tts_service():
    """Test the TTS service"""
    import time
    
    print("🎵 Testing Text-to-Speech Service")
    print("=" * 50)
    
    tts = create_tts_service()
    
    # Test cases
    test_cases = [
        ("Hello, this is a test.", "en"),
        ("नमस्ते, यह एक परीक्षण है।", "hi"),
        ("Hola, esta es una prueba.", "es"),
    ]
    
    for text, lang in test_cases:
        print(f"\n🌍 Testing {lang}: {text}")
        
        result = tts.synthesize_speech(text, lang)
        
        if result['success']:
            print(f"✅ Success!")
            print(f"🔧 Engine: {result['engine']}")
            print(f"📁 Audio: {result['audio_path']}")
            print(f"⏱️ Duration: {result.get('duration', 'Unknown')} seconds")
        else:
            print(f"❌ Failed: {result['error']}")
    
    # Show available voices
    print(f"\n📋 Available Voice Information:")
    voices = tts.get_available_voices()
    for key, value in voices.items():
        if key != 'system_voices':
            print(f"  {key}: {value}")


if __name__ == "__main__":
    test_tts_service()