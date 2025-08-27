"""
Tests for Main Pipeline Integration

This module tests the main SpeechTranslator pipeline and integration
between all components.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.pipeline.main_pipeline import SpeechTranslator, create_speech_translator, quick_translate_audio
from tests.conftest import TEST_CONFIG, MockSpeechRecognizer, MockTranslationService, MockVoiceCloner


class TestSpeechTranslator:
    """Test the main SpeechTranslator class."""
    
    def test_init_default(self):
        """Test SpeechTranslator initialization with default parameters."""
        translator = SpeechTranslator()
        
        assert translator.speech_model is not None
        assert translator.translation_engine is not None
        assert translator.tts_model is not None
        assert translator.device == "auto"
        
        # Components should not be loaded yet
        assert translator.speech_recognizer is None
        assert translator.translation_service is None
        assert translator.voice_cloner is None
    
    def test_init_custom_params(self):
        """Test SpeechTranslator initialization with custom parameters."""
        translator = SpeechTranslator(
            speech_model="small",
            translation_engine="local",
            tts_model="custom_tts",
            device="cpu"
        )
        
        assert translator.speech_model == "small"
        assert translator.translation_engine == "local"
        assert translator.tts_model == "custom_tts"
        assert translator.device == "cpu"
    
    @patch('src.pipeline.main_pipeline.SpeechRecognizer')
    @patch('src.pipeline.main_pipeline.TranslationService')
    @patch('src.pipeline.main_pipeline.VoiceCloner')
    def test_initialize(self, mock_voice_cloner, mock_translation_service, mock_speech_recognizer):
        """Test system initialization with mocked components."""
        # Setup mocks
        mock_speech_recognizer.return_value = MockSpeechRecognizer()
        mock_translation_service.return_value = MockTranslationService()
        mock_voice_cloner.return_value = MockVoiceCloner()
        
        translator = SpeechTranslator()
        translator.initialize(load_models=False)
        
        # Verify components were created
        assert translator.speech_recognizer is not None
        assert translator.translation_service is not None
        assert translator.voice_cloner is not None
    
    @patch('src.pipeline.main_pipeline.SpeechRecognizer')
    @patch('src.pipeline.main_pipeline.TranslationService') 
    @patch('src.pipeline.main_pipeline.VoiceCloner')
    def test_translate_audio_success(self, mock_voice_cloner, mock_translation_service, mock_speech_recognizer, test_audio_file, voice_sample_file, audio_manager):
        """Test successful audio translation."""
        # Setup mocks
        mock_speech_recognizer.return_value = MockSpeechRecognizer()
        mock_translation_service.return_value = MockTranslationService()
        mock_voice_cloner.return_value = MockVoiceCloner()
        
        translator = SpeechTranslator()
        translator.initialize()
        
        output_path = audio_manager.get_temp_dir() / "output.wav"
        
        result = translator.translate_audio(
            input_audio=test_audio_file,
            target_lang="es",
            voice_sample=voice_sample_file,
            output_path=output_path
        )
        
        # Verify result structure
        assert result['success'] is True
        assert 'original_text' in result
        assert 'translated_text' in result
        assert 'source_language' in result
        assert 'target_language' in result
        assert 'processing_time' in result
        assert result['target_language'] == "es"
    
    @patch('src.pipeline.main_pipeline.SpeechRecognizer')
    @patch('src.pipeline.main_pipeline.TranslationService')
    @patch('src.pipeline.main_pipeline.VoiceCloner')
    def test_translate_audio_with_registered_speaker(self, mock_voice_cloner, mock_translation_service, mock_speech_recognizer, test_audio_file, audio_manager):
        """Test audio translation with pre-registered speaker."""
        # Setup mocks
        mock_speech_recognizer.return_value = MockSpeechRecognizer()
        mock_translation_service.return_value = MockTranslationService()
        mock_voice_cloner_instance = MockVoiceCloner()
        mock_voice_cloner.return_value = mock_voice_cloner_instance
        
        translator = SpeechTranslator()
        translator.initialize()
        
        # Register speaker first
        speaker_name = "test_speaker"
        translator.voice_cloner.register_voice(speaker_name, ["dummy_sample.wav"])
        
        output_path = audio_manager.get_temp_dir() / "output.wav"
        
        result = translator.translate_audio(
            input_audio=test_audio_file,
            target_lang="fr",
            speaker_name=speaker_name,
            output_path=output_path
        )
        
        assert result['success'] is True
        assert result['speaker_name'] == speaker_name
    
    @patch('src.pipeline.main_pipeline.SpeechRecognizer')
    @patch('src.pipeline.main_pipeline.TranslationService')
    @patch('src.pipeline.main_pipeline.VoiceCloner')
    def test_translate_text_with_voice(self, mock_voice_cloner, mock_translation_service, mock_speech_recognizer, voice_sample_file, audio_manager):
        """Test text translation with voice generation."""
        # Setup mocks
        mock_translation_service.return_value = MockTranslationService()
        mock_voice_cloner.return_value = MockVoiceCloner()
        
        translator = SpeechTranslator()
        translator.initialize()
        
        output_path = audio_manager.get_temp_dir() / "text_output.wav"
        
        result = translator.translate_text_with_voice(
            text=TEST_CONFIG['test_text'],
            source_lang="en",
            target_lang="es",
            voice_sample=voice_sample_file,
            output_path=output_path
        )
        
        assert result['success'] is True
        assert result['original_text'] == TEST_CONFIG['test_text']
        assert 'translated_text' in result
        assert result['source_language'] == "en"
        assert result['target_language'] == "es"
    
    @patch('src.pipeline.main_pipeline.SpeechRecognizer')
    @patch('src.pipeline.main_pipeline.TranslationService')
    @patch('src.pipeline.main_pipeline.VoiceCloner')
    def test_batch_translate_audio(self, mock_voice_cloner, mock_translation_service, mock_speech_recognizer, batch_audio_files, voice_sample_file, audio_manager):
        """Test batch audio translation."""
        # Setup mocks
        mock_speech_recognizer.return_value = MockSpeechRecognizer()
        mock_translation_service.return_value = MockTranslationService()
        mock_voice_cloner.return_value = MockVoiceCloner()
        
        translator = SpeechTranslator()
        translator.initialize()
        
        output_dir = audio_manager.get_temp_dir() / "batch_output"
        
        result = translator.batch_translate_audio(
            audio_files=batch_audio_files,
            target_lang="es",
            voice_sample=voice_sample_file,
            output_dir=output_dir
        )
        
        assert 'total_files' in result
        assert 'successful' in result
        assert 'failed' in result
        assert 'results' in result
        assert result['total_files'] == len(batch_audio_files)
    
    def test_register_speaker_voice_validation(self):
        """Test speaker voice registration with validation."""
        translator = SpeechTranslator()
        
        # Mock voice cloner
        translator.voice_cloner = MockVoiceCloner()
        
        result = translator.register_speaker_voice(
            speaker_name="test_speaker",
            voice_samples=["sample1.wav", "sample2.wav"],
            validate=False  # Skip validation for mock
        )
        
        assert result['speaker_name'] == "test_speaker"
        assert result['num_samples'] == 2
    
    def test_get_supported_languages(self):
        """Test getting supported languages."""
        translator = SpeechTranslator()
        
        languages = translator.get_supported_languages()
        
        assert isinstance(languages, dict)
        assert 'en' in languages
        assert 'es' in languages
    
    def test_get_system_info(self):
        """Test getting system information."""
        translator = SpeechTranslator()
        
        info = translator.get_system_info()
        
        required_keys = ['configuration', 'components_loaded', 'statistics']
        for key in required_keys:
            assert key in info
        
        assert 'speech_model' in info['configuration']
        assert 'translation_engine' in info['configuration']
        assert 'tts_model' in info['configuration']
    
    @patch('src.pipeline.main_pipeline.SpeechRecognizer')
    @patch('src.pipeline.main_pipeline.TranslationService')
    @patch('src.pipeline.main_pipeline.VoiceCloner')
    def test_session_save_load(self, mock_voice_cloner, mock_translation_service, mock_speech_recognizer, audio_manager):
        """Test session saving and loading."""
        # Setup mocks
        mock_voice_cloner_instance = MockVoiceCloner()
        mock_voice_cloner.return_value = mock_voice_cloner_instance
        
        translator = SpeechTranslator()
        translator.initialize()
        
        # Add some state
        translator.stats['total_processed'] = 5
        translator.voice_cloner.register_voice("test_speaker", ["sample.wav"])
        
        session_path = audio_manager.get_temp_dir() / "session"
        
        # Save session
        translator.save_session(session_path)
        assert session_path.exists()
        
        # Create new translator and load session
        new_translator = SpeechTranslator()
        new_translator.voice_cloner = MockVoiceCloner()  # Mock for loading
        new_translator.load_session(session_path)
        
        # Verify loaded state
        assert new_translator.stats['total_processed'] == 5
    
    def test_progress_callback(self):
        """Test progress callback functionality."""
        progress_messages = []
        
        def progress_callback(message):
            progress_messages.append(message)
        
        translator = SpeechTranslator(progress_callback=progress_callback)
        
        # Test progress update
        translator._update_progress("Test message")
        
        assert len(progress_messages) == 1
        assert progress_messages[0] == "Test message"


class TestSpeechTranslatorErrorHandling:
    """Test error handling in SpeechTranslator."""
    
    @patch('src.pipeline.main_pipeline.SpeechRecognizer')
    def test_translate_audio_speech_recognition_failure(self, mock_speech_recognizer, test_audio_file, voice_sample_file):
        """Test handling of speech recognition failure."""
        # Setup mock to raise exception
        mock_recognizer = MagicMock()
        mock_recognizer.transcribe.side_effect = Exception("Speech recognition failed")
        mock_speech_recognizer.return_value = mock_recognizer
        
        translator = SpeechTranslator()
        translator.speech_recognizer = mock_recognizer
        translator.translation_service = MockTranslationService()
        translator.voice_cloner = MockVoiceCloner()
        
        result = translator.translate_audio(
            input_audio=test_audio_file,
            target_lang="es",
            voice_sample=voice_sample_file
        )
        
        assert result['success'] is False
        assert 'error' in result
        assert "Speech recognition failed" in str(result['error'])
    
    def test_translate_audio_invalid_input(self):
        """Test handling of invalid input audio file."""
        translator = SpeechTranslator()
        translator.speech_recognizer = MockSpeechRecognizer()
        translator.translation_service = MockTranslationService()
        translator.voice_cloner = MockVoiceCloner()
        
        result = translator.translate_audio(
            input_audio="nonexistent_file.wav",
            target_lang="es",
            voice_sample="voice_sample.wav"
        )
        
        assert result['success'] is False
        assert 'error' in result
    
    def test_translate_audio_missing_voice_info(self, test_audio_file):
        """Test handling when neither voice_sample nor speaker_name provided."""
        translator = SpeechTranslator()
        translator.speech_recognizer = MockSpeechRecognizer()
        translator.translation_service = MockTranslationService()
        translator.voice_cloner = MockVoiceCloner()
        
        result = translator.translate_audio(
            input_audio=test_audio_file,
            target_lang="es"
            # No voice_sample or speaker_name provided
        )
        
        assert result['success'] is False
        assert 'error' in result


class TestSpeechTranslatorIntegration:
    """Integration tests for SpeechTranslator."""
    
    @patch('src.pipeline.main_pipeline.SpeechRecognizer')
    @patch('src.pipeline.main_pipeline.TranslationService')
    @patch('src.pipeline.main_pipeline.VoiceCloner')
    def test_full_pipeline_integration(self, mock_voice_cloner, mock_translation_service, mock_speech_recognizer, test_audio_file, voice_sample_file, audio_manager):
        """Test complete pipeline integration with all components."""
        # Setup mocks with realistic responses
        mock_recognizer = MockSpeechRecognizer()
        mock_translator_service = MockTranslationService()
        mock_cloner = MockVoiceCloner()
        
        mock_speech_recognizer.return_value = mock_recognizer
        mock_translation_service.return_value = mock_translator_service
        mock_voice_cloner.return_value = mock_cloner
        
        translator = SpeechTranslator()
        translator.initialize()
        
        output_path = audio_manager.get_temp_dir() / "integrated_output.wav"
        
        # Test full pipeline
        result = translator.translate_audio(
            input_audio=test_audio_file,
            source_lang=None,  # Auto-detect
            target_lang="es",
            voice_sample=voice_sample_file,
            output_path=output_path,
            return_intermediate=True
        )
        
        # Verify complete result
        assert result['success'] is True
        assert 'intermediate_results' in result
        assert 'transcription' in result['intermediate_results']
        assert 'translation' in result['intermediate_results']
        assert 'voice_cloning' in result['intermediate_results']
        
        # Verify pipeline stats were updated
        assert translator.stats['total_processed'] > 0
        assert translator.stats['successful_translations'] > 0


class TestUtilityFunctions:
    """Test utility functions."""
    
    @patch('src.pipeline.main_pipeline.SpeechTranslator')
    def test_create_speech_translator(self, mock_speech_translator):
        """Test create_speech_translator utility function."""
        mock_instance = MagicMock()
        mock_speech_translator.return_value = mock_instance
        
        # Test with default parameters
        translator = create_speech_translator()
        
        mock_speech_translator.assert_called_once()
        mock_instance.initialize.assert_called_once()
    
    @patch('src.pipeline.main_pipeline.SpeechTranslator')
    def test_create_speech_translator_no_init(self, mock_speech_translator):
        """Test create_speech_translator without initialization."""
        mock_instance = MagicMock()
        mock_speech_translator.return_value = mock_instance
        
        translator = create_speech_translator(initialize=False)
        
        mock_speech_translator.assert_called_once()
        mock_instance.initialize.assert_not_called()
    
    @patch('src.pipeline.main_pipeline.create_speech_translator')
    def test_quick_translate_audio(self, mock_create_translator, audio_manager):
        """Test quick_translate_audio utility function."""
        # Setup mock translator
        mock_translator = MagicMock()
        mock_translator.translate_audio.return_value = {
            'success': True,
            'output_audio': 'output.wav'
        }
        mock_create_translator.return_value = mock_translator
        
        input_audio = "input.wav"
        voice_sample = "voice.wav"
        
        result = quick_translate_audio(
            input_audio=input_audio,
            voice_sample=voice_sample,
            target_lang="es"
        )
        
        assert result == 'output.wav'
        mock_translator.translate_audio.assert_called_once()
    
    @patch('src.pipeline.main_pipeline.create_speech_translator')
    def test_quick_translate_audio_failure(self, mock_create_translator):
        """Test quick_translate_audio with failure."""
        # Setup mock translator to return failure
        mock_translator = MagicMock()
        mock_translator.translate_audio.return_value = {
            'success': False,
            'error': 'Translation failed'
        }
        mock_create_translator.return_value = mock_translator
        
        with pytest.raises(RuntimeError, match="Translation failed"):
            quick_translate_audio(
                input_audio="input.wav",
                voice_sample="voice.wav",
                target_lang="es"
            )


# Performance and stress tests
@pytest.mark.slow
class TestSpeechTranslatorPerformance:
    """Performance tests for SpeechTranslator."""
    
    @patch('src.pipeline.main_pipeline.SpeechRecognizer')
    @patch('src.pipeline.main_pipeline.TranslationService')
    @patch('src.pipeline.main_pipeline.VoiceCloner')
    def test_multiple_translations_performance(self, mock_voice_cloner, mock_translation_service, mock_speech_recognizer, test_audio_file, voice_sample_file):
        """Test performance with multiple sequential translations."""
        # Setup mocks
        mock_speech_recognizer.return_value = MockSpeechRecognizer()
        mock_translation_service.return_value = MockTranslationService()
        mock_voice_cloner.return_value = MockVoiceCloner()
        
        translator = SpeechTranslator()
        translator.initialize()
        
        import time
        start_time = time.time()
        
        # Perform multiple translations
        for i in range(5):
            result = translator.translate_audio(
                input_audio=test_audio_file,
                target_lang="es",
                voice_sample=voice_sample_file
            )
            assert result['success'] is True
        
        total_time = time.time() - start_time
        
        # Should complete multiple translations in reasonable time
        assert total_time < 30.0  # 30 seconds for 5 translations with mocks
        
        # Verify stats
        assert translator.stats['total_processed'] == 5
        assert translator.stats['successful_translations'] == 5


if __name__ == "__main__":
    pytest.main([__file__])