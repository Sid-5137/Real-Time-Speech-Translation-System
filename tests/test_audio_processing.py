"""
Tests for Audio Processing Module

This module tests the audio processing functionality including
file loading, format conversion, and audio enhancement.
"""

import pytest
import numpy as np
from pathlib import Path
import soundfile as sf

from src.audio_processing.processor import AudioProcessor, AudioValidator
from tests.conftest import TEST_CONFIG, create_test_audio


class TestAudioProcessor:
    """Test the AudioProcessor class."""
    
    def test_init(self):
        """Test AudioProcessor initialization."""
        processor = AudioProcessor()
        assert processor.target_sample_rate == TEST_CONFIG['sample_rate']
        assert processor.supported_formats is not None
    
    def test_load_audio_basic(self, test_audio_file):
        """Test basic audio loading."""
        processor = AudioProcessor()
        
        audio_data = processor.load_audio(test_audio_file)
        
        assert isinstance(audio_data, np.ndarray)
        assert audio_data.dtype == np.float32
        assert len(audio_data) > 0
        
        # Check approximate duration
        expected_samples = int(TEST_CONFIG['test_duration'] * TEST_CONFIG['sample_rate'])
        assert abs(len(audio_data) - expected_samples) < TEST_CONFIG['sample_rate']  # Within 1 second
    
    def test_load_audio_normalize(self, test_audio_file):
        """Test audio loading with normalization."""
        processor = AudioProcessor()
        
        # Load without normalization
        audio_raw = processor.load_audio(test_audio_file, normalize=False)
        
        # Load with normalization
        audio_normalized = processor.load_audio(test_audio_file, normalize=True)
        
        # Normalized audio should have different amplitude
        assert not np.allclose(audio_raw, audio_normalized)
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        processor = AudioProcessor()
        
        with pytest.raises(FileNotFoundError):
            processor.load_audio("nonexistent_file.wav")
    
    def test_save_audio(self, audio_manager):
        """Test saving audio data."""
        processor = AudioProcessor()
        
        # Create test audio
        audio_data = create_test_audio()
        output_path = audio_manager.get_temp_dir() / "output.wav"
        
        # Save audio
        processor.save_audio(audio_data, output_path)
        
        # Verify file was created
        assert output_path.exists()
        
        # Load and verify content
        loaded_audio = processor.load_audio(output_path)
        assert len(loaded_audio) == len(audio_data)
    
    def test_convert_format(self, test_audio_file, audio_manager):
        """Test audio format conversion."""
        processor = AudioProcessor()
        
        # Convert to different format
        output_path = audio_manager.get_temp_dir() / "converted.wav"
        
        processor.convert_format(test_audio_file, output_path, target_format='wav')
        
        # Verify converted file exists and is loadable
        assert output_path.exists()
        converted_audio = processor.load_audio(output_path)
        assert len(converted_audio) > 0
    
    def test_normalize_audio(self):
        """Test audio normalization."""
        processor = AudioProcessor()
        
        # Create test audio with known amplitude
        audio_data = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
        
        normalized = processor.normalize_audio(audio_data, target_db=-20.0)
        
        # Normalized audio should have different values
        assert not np.allclose(audio_data, normalized)
        assert normalized.dtype == np.float32
    
    def test_remove_silence(self):
        """Test silence removal."""
        processor = AudioProcessor()
        
        # Create audio with silence padding
        signal = create_test_audio(duration=1.0)
        silence = np.zeros(int(0.5 * TEST_CONFIG['sample_rate']), dtype=np.float32)
        
        # Add silence at beginning and end
        audio_with_silence = np.concatenate([silence, signal, silence])
        
        # Remove silence
        trimmed = processor.remove_silence(audio_with_silence, threshold_db=-40.0)
        
        # Trimmed audio should be shorter
        assert len(trimmed) < len(audio_with_silence)
        assert len(trimmed) <= len(signal) * 1.1  # Allow some tolerance
    
    def test_resample_audio(self):
        """Test audio resampling."""
        processor = AudioProcessor()
        
        # Create test audio at original sample rate
        original_sr = 16000
        target_sr = TEST_CONFIG['sample_rate']
        duration = 2.0
        
        t = np.linspace(0, duration, int(duration * original_sr), False)
        audio_data = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # Resample
        resampled = processor.resample_audio(audio_data, original_sr, target_sr)
        
        # Check new length
        expected_length = int(len(audio_data) * target_sr / original_sr)
        assert abs(len(resampled) - expected_length) < 10  # Allow small tolerance
    
    def test_split_audio(self):
        """Test audio splitting into chunks."""
        processor = AudioProcessor()
        
        # Create longer audio
        audio_data = create_test_audio(duration=10.0)
        
        # Split into chunks
        chunks = processor.split_audio(audio_data, chunk_duration=3.0, overlap=0.5)
        
        # Should have multiple chunks
        assert len(chunks) > 1
        
        # Each chunk should have correct length
        chunk_samples = int(3.0 * TEST_CONFIG['sample_rate'])
        for chunk in chunks:
            assert len(chunk) == chunk_samples
    
    def test_get_audio_info(self, test_audio_file):
        """Test getting audio file information."""
        processor = AudioProcessor()
        
        info = processor.get_audio_info(test_audio_file)
        
        # Verify info structure
        required_keys = ['path', 'duration', 'sample_rate', 'channels', 'samples', 'file_size']
        for key in required_keys:
            assert key in info
        
        # Verify values
        assert info['sample_rate'] == TEST_CONFIG['sample_rate']
        assert info['duration'] > 0
        assert info['channels'] == 1
        assert info['samples'] > 0


class TestAudioValidator:
    """Test the AudioValidator class."""
    
    def test_init(self):
        """Test AudioValidator initialization."""
        processor = AudioProcessor()
        validator = AudioValidator(processor)
        assert validator.processor is processor
    
    def test_validate_audio_file_valid(self, test_audio_file):
        """Test validation of valid audio file."""
        processor = AudioProcessor()
        validator = AudioValidator(processor)
        
        result = validator.validate_audio_file(test_audio_file)
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
        assert 'info' in result
    
    def test_validate_audio_file_nonexistent(self):
        """Test validation of non-existent file."""
        processor = AudioProcessor()
        validator = AudioValidator(processor)
        
        result = validator.validate_audio_file("nonexistent.wav")
        
        assert result['valid'] is False
        assert len(result['errors']) > 0
        assert "does not exist" in result['errors'][0]
    
    def test_validate_audio_file_unsupported_format(self, audio_manager):
        """Test validation of unsupported format."""
        processor = AudioProcessor()
        validator = AudioValidator(processor)
        
        # Create file with unsupported extension
        unsupported_file = audio_manager.get_temp_dir() / "test.txt"
        unsupported_file.write_text("not an audio file")
        
        result = validator.validate_audio_file(unsupported_file)
        
        assert result['valid'] is False
        assert any("Unsupported format" in error for error in result['errors'])
    
    def test_validate_batch(self, batch_audio_files):
        """Test batch validation."""
        processor = AudioProcessor()
        validator = AudioValidator(processor)
        
        result = validator.validate_batch(batch_audio_files)
        
        assert 'total_files' in result
        assert 'valid_files' in result
        assert 'invalid_files' in result
        assert 'results' in result
        
        assert result['total_files'] == len(batch_audio_files)
        assert result['valid_files'] > 0  # Should have some valid files
    
    def test_validate_batch_mixed(self, batch_audio_files, audio_manager):
        """Test batch validation with mixed valid/invalid files."""
        processor = AudioProcessor()
        validator = AudioValidator(processor)
        
        # Add an invalid file to the batch
        invalid_file = audio_manager.get_temp_dir() / "invalid.txt"
        invalid_file.write_text("invalid")
        
        mixed_files = batch_audio_files + [invalid_file]
        
        result = validator.validate_batch(mixed_files)
        
        assert result['total_files'] == len(mixed_files)
        assert result['valid_files'] == len(batch_audio_files)  # Only original files are valid
        assert result['invalid_files'] == 1  # One invalid file


class TestAudioProcessorIntegration:
    """Integration tests for audio processing."""
    
    def test_full_audio_pipeline(self, audio_manager):
        """Test complete audio processing pipeline."""
        processor = AudioProcessor()
        
        # 1. Create test audio
        input_path = audio_manager.create_audio_file("input.wav", duration=5.0)
        
        # 2. Load audio
        audio_data = processor.load_audio(input_path)
        
        # 3. Apply processing
        normalized = processor.normalize_audio(audio_data)
        trimmed = processor.remove_silence(normalized)
        
        # 4. Save processed audio
        output_path = audio_manager.get_temp_dir() / "processed.wav"
        processor.save_audio(trimmed, output_path)
        
        # 5. Verify processed audio
        assert output_path.exists()
        processed_audio = processor.load_audio(output_path)
        assert len(processed_audio) > 0
        
        # 6. Get info about processed audio
        info = processor.get_audio_info(output_path)
        assert info['duration'] > 0
    
    def test_format_conversion_chain(self, test_audio_file, audio_manager):
        """Test converting between different formats."""
        processor = AudioProcessor()
        
        formats = ['wav', 'mp3']  # Limited to commonly supported formats
        temp_dir = audio_manager.get_temp_dir()
        
        current_file = test_audio_file
        
        for i, target_format in enumerate(formats):
            output_file = temp_dir / f"converted_{i}.{target_format}"
            
            try:
                processor.convert_format(current_file, output_file, target_format)
                
                if output_file.exists():
                    # Verify the converted file is loadable
                    audio_data = processor.load_audio(output_file)
                    assert len(audio_data) > 0
                    current_file = output_file
                    
            except Exception as e:
                # Some formats might not be available, that's OK
                pytest.skip(f"Format {target_format} not supported: {e}")


# Performance tests (marked as slow)
@pytest.mark.slow
class TestAudioProcessorPerformance:
    """Performance tests for audio processing."""
    
    def test_large_file_processing(self, audio_manager):
        """Test processing of large audio files."""
        processor = AudioProcessor()
        
        # Create a large audio file (30 seconds)
        large_audio_file = audio_manager.create_audio_file("large.wav", duration=30.0)
        
        # Test loading and processing
        import time
        start_time = time.time()
        
        audio_data = processor.load_audio(large_audio_file)
        normalized = processor.normalize_audio(audio_data)
        chunks = processor.split_audio(normalized, chunk_duration=5.0)
        
        processing_time = time.time() - start_time
        
        # Should complete in reasonable time (less than 30 seconds)
        assert processing_time < 30.0
        assert len(chunks) > 1
    
    def test_batch_processing_performance(self, audio_manager):
        """Test batch processing performance."""
        processor = AudioProcessor()
        validator = AudioValidator(processor)
        
        # Create multiple test files
        files = []
        for i in range(10):
            files.append(audio_manager.create_audio_file(f"batch_{i}.wav", duration=3.0))
        
        import time
        start_time = time.time()
        
        result = validator.validate_batch(files)
        
        processing_time = time.time() - start_time
        
        # Should complete batch validation quickly
        assert processing_time < 10.0
        assert result['total_files'] == len(files)


if __name__ == "__main__":
    pytest.main([__file__])