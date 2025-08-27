"""
Main Pipeline Module

This module provides the main SpeechTranslator class that orchestrates
the entire speech translation workflow with voice cloning.
"""

import logging
import time
from typing import Dict, List, Optional, Union, Any, Callable
from pathlib import Path
import json

from ..speech_recognition.whisper_recognizer import SpeechRecognizer, create_speech_recognizer
from ..translation.translator import TranslationService, create_translation_service
from ..voice_cloning.voice_cloner import VoiceCloner, create_voice_cloner
from ..audio_processing.processor import AudioProcessor, AudioValidator
from ..config import (
    WHISPER_MODEL_SIZE, DEFAULT_TRANSLATION_SERVICE, TTS_MODEL, 
    SUPPORTED_LANGUAGES, SAMPLE_RATE
)


class SpeechTranslator:
    """Main speech translation system with voice cloning."""
    
    def __init__(
        self,
        speech_model: str = WHISPER_MODEL_SIZE,
        translation_engine: str = DEFAULT_TRANSLATION_SERVICE,
        tts_model: str = TTS_MODEL,
        device: str = "auto",
        progress_callback: Optional[Callable] = None
    ):
        """
        Initialize the speech translator.
        
        Args:
            speech_model: Whisper model size for speech recognition
            translation_engine: Translation engine ('google' or 'local')
            tts_model: TTS model for voice cloning
            device: Device to run models on
            progress_callback: Optional callback for progress updates
        """
        self.speech_model = speech_model
        self.translation_engine = translation_engine
        self.tts_model = tts_model
        self.device = device
        self.progress_callback = progress_callback
        
        # Initialize components
        self.speech_recognizer = None
        self.translation_service = None
        self.voice_cloner = None
        self.audio_processor = AudioProcessor()
        self.audio_validator = AudioValidator(self.audio_processor)
        
        self.logger = logging.getLogger(__name__)
        
        # Processing statistics
        self.stats = {
            'total_processed': 0,
            'successful_translations': 0,
            'failed_translations': 0,
            'total_processing_time': 0.0
        }
        
    def initialize(self, load_models: bool = True) -> None:
        """
        Initialize all components.
        
        Args:
            load_models: Whether to load models immediately
        """
        try:
            self.logger.info("Initializing Speech Translation System...")
            
            # Initialize speech recognizer
            self._update_progress("Loading speech recognition model...")
            self.speech_recognizer = SpeechRecognizer(
                model_size=self.speech_model,
                device=self.device
            )
            if load_models:
                self.speech_recognizer.load_model()
            
            # Initialize translation service
            self._update_progress("Initializing translation service...")
            self.translation_service = TranslationService(
                primary_engine=self.translation_engine,
                fallback_engine="google" if self.translation_engine != "google" else None
            )
            
            # Initialize voice cloner
            self._update_progress("Loading voice cloning model...")
            self.voice_cloner = VoiceCloner(
                model_name=self.tts_model,
                device=self.device
            )
            if load_models:
                self.voice_cloner.load_model()
            
            self._update_progress("Initialization complete!")
            self.logger.info("Speech Translation System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            raise RuntimeError(f"System initialization failed: {str(e)}")
    
    def translate_audio(
        self,
        input_audio: Union[str, Path],
        source_lang: Optional[str] = None,
        target_lang: str = "en",
        voice_sample: Optional[Union[str, Path]] = None,
        speaker_name: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None,
        return_intermediate: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Translate audio with voice cloning.
        
        Args:
            input_audio: Path to input audio file
            source_lang: Source language (auto-detected if None)
            target_lang: Target language code
            voice_sample: Path to voice sample for cloning
            speaker_name: Name of registered speaker (alternative to voice_sample)
            output_path: Path for output audio file
            return_intermediate: Whether to return intermediate results
            **kwargs: Additional parameters for each component
            
        Returns:
            Dictionary with translation results and generated audio
        """
        if not self.speech_recognizer or not self.translation_service or not self.voice_cloner:
            self.initialize()
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting audio translation: {input_audio}")
            
            # Step 1: Validate input audio
            self._update_progress("Validating input audio...")
            validation = self.audio_validator.validate_audio_file(input_audio)
            if not validation['valid']:
                raise ValueError(f"Invalid audio file: {validation['errors']}")
            
            # Step 2: Speech Recognition
            self._update_progress("Converting speech to text...")
            transcription_result = self.speech_recognizer.transcribe(
                input_audio, 
                language=source_lang,
                **kwargs.get('speech_recognition', {})
            )
            
            original_text = transcription_result['text']
            detected_language = transcription_result['language']
            
            self.logger.info(f"Transcribed text: {original_text[:100]}...")
            self.logger.info(f"Detected language: {detected_language}")
            
            # Step 3: Translation
            self._update_progress("Translating text...")
            translation_result = self.translation_service.translate(
                text=original_text,
                source_lang=detected_language,
                target_lang=target_lang,
                **kwargs.get('translation', {})
            )
            
            translated_text = translation_result['translated_text']
            self.logger.info(f"Translated text: {translated_text[:100]}...")
            
            # Step 4: Voice Cloning Setup
            if voice_sample and not speaker_name:
                # Register temporary speaker
                speaker_name = f"temp_speaker_{int(time.time())}"
                self._update_progress("Registering voice sample...")
                self.voice_cloner.register_voice(
                    speaker_name, 
                    [voice_sample],
                    **kwargs.get('voice_registration', {})
                )
            elif not speaker_name:
                raise ValueError("Either voice_sample or speaker_name must be provided")
            
            # Step 5: Voice Cloning
            self._update_progress("Generating speech with cloned voice...")
            voice_result = self.voice_cloner.clone_voice(
                text=translated_text,
                speaker_name=speaker_name,
                language=target_lang,
                output_path=output_path,
                **kwargs.get('voice_cloning', {})
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update statistics
            self.stats['total_processed'] += 1
            self.stats['successful_translations'] += 1
            self.stats['total_processing_time'] += processing_time
            
            # Prepare results
            result = {
                'success': True,
                'input_audio': str(input_audio),
                'output_audio': voice_result['output_path'],
                'original_text': original_text,
                'translated_text': translated_text,
                'source_language': detected_language,
                'target_language': target_lang,
                'speaker_name': speaker_name,
                'processing_time': processing_time,
                'audio_duration': voice_result['duration'],
                'model_info': {
                    'speech_model': self.speech_model,
                    'translation_engine': self.translation_engine,
                    'tts_model': self.tts_model
                }
            }
            
            # Add intermediate results if requested
            if return_intermediate:
                result['intermediate_results'] = {
                    'transcription': transcription_result,
                    'translation': translation_result,
                    'voice_cloning': voice_result
                }
            
            self._update_progress("Translation completed successfully!")
            self.logger.info(f"Audio translation completed in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.stats['failed_translations'] += 1
            self.logger.error(f"Audio translation failed: {str(e)}")
            
            error_result = {
                'success': False,
                'error': str(e),
                'input_audio': str(input_audio),
                'processing_time': time.time() - start_time
            }
            
            return error_result
    
    def translate_text_with_voice(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        voice_sample: Optional[Union[str, Path]] = None,
        speaker_name: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Translate text and generate speech with cloned voice.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            voice_sample: Path to voice sample for cloning
            speaker_name: Name of registered speaker
            output_path: Path for output audio file
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with translation and voice cloning results
        """
        if not self.translation_service or not self.voice_cloner:
            self.initialize()
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting text translation with voice: {text[:50]}...")
            
            # Step 1: Translation
            self._update_progress("Translating text...")
            translation_result = self.translation_service.translate(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                **kwargs.get('translation', {})
            )
            
            translated_text = translation_result['translated_text']
            
            # Step 2: Voice Setup
            if voice_sample and not speaker_name:
                speaker_name = f"temp_speaker_{int(time.time())}"
                self.voice_cloner.register_voice(speaker_name, [voice_sample])
            elif not speaker_name:
                raise ValueError("Either voice_sample or speaker_name must be provided")
            
            # Step 3: Voice Generation
            self._update_progress("Generating speech...")
            voice_result = self.voice_cloner.clone_voice(
                text=translated_text,
                speaker_name=speaker_name,
                language=target_lang,
                output_path=output_path,
                **kwargs.get('voice_cloning', {})
            )
            
            processing_time = time.time() - start_time
            
            result = {
                'success': True,
                'original_text': text,
                'translated_text': translated_text,
                'source_language': source_lang,
                'target_language': target_lang,
                'speaker_name': speaker_name,
                'output_audio': voice_result['output_path'],
                'processing_time': processing_time,
                'audio_duration': voice_result['duration']
            }
            
            self._update_progress("Text translation completed!")
            return result
            
        except Exception as e:
            self.logger.error(f"Text translation with voice failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'original_text': text,
                'processing_time': time.time() - start_time
            }
    
    def batch_translate_audio(
        self,
        audio_files: List[Union[str, Path]],
        source_lang: Optional[str] = None,
        target_lang: str = "en",
        voice_sample: Optional[Union[str, Path]] = None,
        speaker_name: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Batch translate multiple audio files.
        
        Args:
            audio_files: List of audio file paths
            source_lang: Source language (auto-detected if None)
            target_lang: Target language code
            voice_sample: Voice sample for cloning
            speaker_name: Registered speaker name
            output_dir: Output directory for generated files
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with batch processing results
        """
        if not self.speech_recognizer or not self.translation_service or not self.voice_cloner:
            self.initialize()
        
        results = []
        failed_files = []
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup voice if provided
        if voice_sample and not speaker_name:
            speaker_name = f"batch_speaker_{int(time.time())}"
            self.voice_cloner.register_voice(speaker_name, [voice_sample])
        
        self.logger.info(f"Starting batch translation: {len(audio_files)} files")
        
        for i, audio_file in enumerate(audio_files, 1):
            try:
                self._update_progress(f"Processing file {i}/{len(audio_files)}: {Path(audio_file).name}")
                
                # Generate output path
                output_path = None
                if output_dir:
                    filename = Path(audio_file).stem
                    output_path = output_dir / f"{filename}_translated.wav"
                
                result = self.translate_audio(
                    input_audio=audio_file,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    speaker_name=speaker_name,
                    output_path=output_path,
                    **kwargs
                )
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to process {audio_file}: {str(e)}")
                failed_files.append({
                    'file': str(audio_file),
                    'error': str(e)
                })
        
        batch_result = {
            'total_files': len(audio_files),
            'successful': len(results),
            'failed': len(failed_files),
            'results': results,
            'failed_files': failed_files,
            'speaker_name': speaker_name,
            'target_language': target_lang
        }
        
        self.logger.info(f"Batch processing completed. Success: {batch_result['successful']}, "
                        f"Failed: {batch_result['failed']}")
        
        return batch_result
    
    def register_speaker_voice(
        self,
        speaker_name: str,
        voice_samples: List[Union[str, Path]],
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Register a speaker voice for reuse.
        
        Args:
            speaker_name: Unique speaker identifier
            voice_samples: List of voice sample file paths
            validate: Whether to validate samples
            
        Returns:
            Registration result
        """
        if not self.voice_cloner:
            self.voice_cloner = VoiceCloner(model_name=self.tts_model, device=self.device)
            self.voice_cloner.load_model()
        
        return self.voice_cloner.register_voice(speaker_name, voice_samples, validate)
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages."""
        return SUPPORTED_LANGUAGES
    
    def get_registered_speakers(self) -> List[str]:
        """Get list of registered speakers."""
        if not self.voice_cloner:
            return []
        return self.voice_cloner.get_registered_speakers()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and status."""
        info = {
            'configuration': {
                'speech_model': self.speech_model,
                'translation_engine': self.translation_engine,
                'tts_model': self.tts_model,
                'device': self.device
            },
            'components_loaded': {
                'speech_recognizer': self.speech_recognizer is not None,
                'translation_service': self.translation_service is not None,
                'voice_cloner': self.voice_cloner is not None
            },
            'statistics': self.stats.copy(),
            'supported_languages': len(SUPPORTED_LANGUAGES),
            'registered_speakers': len(self.get_registered_speakers())
        }
        
        # Add component-specific info if loaded
        if self.speech_recognizer:
            info['speech_recognizer_info'] = self.speech_recognizer.get_model_info()
        
        if self.translation_service:
            info['available_translation_engines'] = self.translation_service.get_available_engines()
        
        if self.voice_cloner:
            info['voice_cloner_info'] = self.voice_cloner.get_model_info()
        
        return info
    
    def save_session(self, session_path: Union[str, Path]) -> None:
        """Save current session including registered speakers."""
        session_path = Path(session_path)
        session_path.mkdir(parents=True, exist_ok=True)
        
        # Save system configuration
        config_file = session_path / "session_config.json"
        config = {
            'speech_model': self.speech_model,
            'translation_engine': self.translation_engine,
            'tts_model': self.tts_model,
            'device': self.device,
            'statistics': self.stats
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save speaker data if voice cloner is loaded
        if self.voice_cloner:
            self.voice_cloner.save_speaker_data(session_path / "speakers")
        
        self.logger.info(f"Session saved to: {session_path}")
    
    def load_session(self, session_path: Union[str, Path]) -> None:
        """Load previous session."""
        session_path = Path(session_path)
        
        # Load configuration
        config_file = session_path / "session_config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            self.stats.update(config.get('statistics', {}))
        
        # Load speaker data
        speakers_dir = session_path / "speakers"
        if speakers_dir.exists() and self.voice_cloner:
            self.voice_cloner.load_speaker_data(speakers_dir)
        
        self.logger.info(f"Session loaded from: {session_path}")
    
    def _update_progress(self, message: str) -> None:
        """Update progress via callback if available."""
        if self.progress_callback:
            self.progress_callback(message)
        self.logger.debug(message)


# Convenience functions
def create_speech_translator(
    speech_model: str = WHISPER_MODEL_SIZE,
    translation_engine: str = DEFAULT_TRANSLATION_SERVICE,
    tts_model: str = TTS_MODEL,
    device: str = "auto",
    initialize: bool = True
) -> SpeechTranslator:
    """
    Create and optionally initialize a speech translator.
    
    Args:
        speech_model: Whisper model size
        translation_engine: Translation engine to use
        tts_model: TTS model for voice cloning
        device: Device to run on
        initialize: Whether to initialize immediately
        
    Returns:
        SpeechTranslator instance
    """
    translator = SpeechTranslator(
        speech_model=speech_model,
        translation_engine=translation_engine,
        tts_model=tts_model,
        device=device
    )
    
    if initialize:
        translator.initialize()
    
    return translator


def quick_translate_audio(
    input_audio: Union[str, Path],
    voice_sample: Union[str, Path],
    target_lang: str = "en",
    output_path: Optional[Union[str, Path]] = None
) -> str:
    """
    Quick audio translation for simple use cases.
    
    Args:
        input_audio: Input audio file
        voice_sample: Voice sample for cloning
        target_lang: Target language
        output_path: Output file path
        
    Returns:
        Path to generated audio file
    """
    translator = create_speech_translator()
    
    result = translator.translate_audio(
        input_audio=input_audio,
        target_lang=target_lang,
        voice_sample=voice_sample,
        output_path=output_path
    )
    
    if result['success']:
        return result['output_audio']
    else:
        raise RuntimeError(f"Translation failed: {result['error']}")