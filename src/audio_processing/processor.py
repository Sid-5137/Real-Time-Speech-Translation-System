"""
Audio Processing Module

This module provides comprehensive audio processing capabilities including
format conversion, quality enhancement, and preprocessing for the speech
translation system.
"""

import os
import logging
from typing import Optional, Union, Tuple, List
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from scipy import signal
import torch
import torchaudio

from ..config import SAMPLE_RATE, MAX_AUDIO_DURATION, AUDIO_FORMATS


class AudioProcessor:
    """Handles audio file processing, conversion, and enhancement."""
    
    def __init__(self, target_sample_rate: int = SAMPLE_RATE):
        """
        Initialize the audio processor.
        
        Args:
            target_sample_rate: Target sample rate for processing
        """
        self.target_sample_rate = target_sample_rate
        self.max_duration = MAX_AUDIO_DURATION
        self.supported_formats = AUDIO_FORMATS
        
        self.logger = logging.getLogger(__name__)
        
    def load_audio(
        self, 
        audio_path: Union[str, Path], 
        normalize: bool = True,
        mono: bool = True
    ) -> np.ndarray:
        """
        Load audio file and convert to target format.
        
        Args:
            audio_path: Path to audio file
            normalize: Whether to normalize audio amplitude
            mono: Whether to convert to mono
            
        Returns:
            Audio data as numpy array
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if audio_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported audio format: {audio_path.suffix}")
        
        try:
            self.logger.debug(f"Loading audio: {audio_path}")
            
            # Load audio using librosa (handles most formats)
            audio_data, sample_rate = librosa.load(
                str(audio_path),
                sr=self.target_sample_rate,
                mono=mono,
                dtype=np.float32
            )
            
            # Validate duration
            duration = len(audio_data) / self.target_sample_rate
            if duration > self.max_duration:
                self.logger.warning(f"Audio duration ({duration:.1f}s) exceeds maximum "
                                  f"({self.max_duration}s). Truncating.")
                audio_data = audio_data[:int(self.max_duration * self.target_sample_rate)]
            
            # Normalize amplitude if requested
            if normalize:
                audio_data = self.normalize_audio(audio_data)
            
            self.logger.debug(f"Loaded audio: duration={duration:.2f}s, "
                            f"sample_rate={self.target_sample_rate}, shape={audio_data.shape}")
            
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Failed to load audio {audio_path}: {str(e)}")
            raise RuntimeError(f"Audio loading failed: {str(e)}")
    
    def save_audio(
        self, 
        audio_data: np.ndarray, 
        output_path: Union[str, Path], 
        sample_rate: Optional[int] = None,
        format: Optional[str] = None
    ) -> None:
        """
        Save audio data to file.
        
        Args:
            audio_data: Audio data as numpy array
            output_path: Output file path
            sample_rate: Sample rate (uses target_sample_rate if None)
            format: Audio format (inferred from extension if None)
        """
        output_path = Path(output_path)
        sample_rate = sample_rate or self.target_sample_rate
        
        try:
            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine format from extension if not specified
            if format is None:
                format = output_path.suffix.lower().lstrip('.')
            
            # Ensure audio data is in correct range for format
            if format in ['wav', 'flac']:
                # For lossless formats, keep full precision
                sf.write(str(output_path), audio_data, sample_rate, format=format.upper())
            else:
                # For compressed formats, use pydub
                self._save_with_pydub(audio_data, output_path, sample_rate, format)
            
            self.logger.debug(f"Saved audio to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save audio to {output_path}: {str(e)}")
            raise RuntimeError(f"Audio saving failed: {str(e)}")
    
    def _save_with_pydub(
        self, 
        audio_data: np.ndarray, 
        output_path: Path, 
        sample_rate: int,
        format: str
    ) -> None:
        """Save audio using pydub for compressed formats."""
        # Convert to 16-bit PCM for pydub
        audio_16bit = (audio_data * 32767).astype(np.int16)
        
        # Create AudioSegment
        audio_segment = AudioSegment(
            audio_16bit.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
        
        # Export with format-specific settings
        export_params = {}
        if format == 'mp3':
            export_params['bitrate'] = '192k'
        elif format == 'ogg':
            export_params['codec'] = 'libvorbis'
        
        audio_segment.export(str(output_path), format=format, **export_params)
    
    def convert_format(
        self, 
        input_path: Union[str, Path], 
        output_path: Union[str, Path],
        target_format: str = 'wav'
    ) -> None:
        """
        Convert audio file to different format.
        
        Args:
            input_path: Input audio file path
            output_path: Output audio file path  
            target_format: Target audio format
        """
        audio_data = self.load_audio(input_path)
        
        # Update output path extension if needed
        output_path = Path(output_path)
        if output_path.suffix.lower() != f'.{target_format}':
            output_path = output_path.with_suffix(f'.{target_format}')
        
        self.save_audio(audio_data, output_path, format=target_format)
        self.logger.info(f"Converted {input_path} to {output_path} ({target_format})")
    
    def normalize_audio(self, audio_data: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """
        Normalize audio amplitude.
        
        Args:
            audio_data: Input audio data
            target_db: Target RMS level in dB
            
        Returns:
            Normalized audio data
        """
        # Calculate RMS
        rms = np.sqrt(np.mean(audio_data ** 2))
        
        if rms > 0:
            # Convert target dB to linear scale
            target_linear = 10 ** (target_db / 20.0)
            
            # Calculate scaling factor
            scale_factor = target_linear / rms
            
            # Apply scaling with clipping prevention
            normalized = audio_data * scale_factor
            normalized = np.clip(normalized, -0.95, 0.95)
            
            return normalized
        
        return audio_data
    
    def remove_silence(
        self, 
        audio_data: np.ndarray, 
        threshold_db: float = -40.0,
        frame_length: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        """
        Remove silence from audio.
        
        Args:
            audio_data: Input audio data
            threshold_db: Silence threshold in dB
            frame_length: Frame length for analysis
            hop_length: Hop length for analysis
            
        Returns:
            Audio data with silence removed
        """
        # Calculate frame-wise energy
        frames = librosa.util.frame(
            audio_data, 
            frame_length=frame_length, 
            hop_length=hop_length
        )
        energy = np.sum(frames ** 2, axis=0)
        
        # Convert to dB
        energy_db = librosa.power_to_db(energy)
        
        # Find non-silent frames
        non_silent = energy_db > threshold_db
        
        if not np.any(non_silent):
            self.logger.warning("No non-silent frames found, returning original audio")
            return audio_data
        
        # Convert frame indices to sample indices
        start_frame = np.argmax(non_silent)
        end_frame = len(non_silent) - np.argmax(non_silent[::-1]) - 1
        
        start_sample = start_frame * hop_length
        end_sample = min(len(audio_data), (end_frame + 1) * hop_length + frame_length)
        
        return audio_data[start_sample:end_sample]
    
    def apply_noise_reduction(
        self, 
        audio_data: np.ndarray, 
        noise_factor: float = 0.1
    ) -> np.ndarray:
        """
        Apply basic noise reduction using spectral subtraction.
        
        Args:
            audio_data: Input audio data
            noise_factor: Noise reduction factor (0.0 to 1.0)
            
        Returns:
            Noise-reduced audio data
        """
        # Compute STFT
        stft = librosa.stft(audio_data)
        magnitude, phase = np.abs(stft), np.angle(stft)
        
        # Estimate noise from first few frames (assume silence)
        noise_frames = min(10, magnitude.shape[1] // 4)
        noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # Apply spectral subtraction
        magnitude_clean = magnitude - (noise_factor * noise_spectrum)
        magnitude_clean = np.maximum(magnitude_clean, 0.1 * magnitude)
        
        # Reconstruct signal
        stft_clean = magnitude_clean * np.exp(1j * phase)
        audio_clean = librosa.istft(stft_clean)
        
        return audio_clean
    
    def resample_audio(
        self, 
        audio_data: np.ndarray, 
        original_sr: int, 
        target_sr: int
    ) -> np.ndarray:
        """
        Resample audio to different sample rate.
        
        Args:
            audio_data: Input audio data
            original_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio data
        """
        if original_sr == target_sr:
            return audio_data
        
        return librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr)
    
    def split_audio(
        self, 
        audio_data: np.ndarray, 
        chunk_duration: float = 30.0,
        overlap: float = 0.5
    ) -> List[np.ndarray]:
        """
        Split audio into overlapping chunks.
        
        Args:
            audio_data: Input audio data
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks (0.0 to 1.0)
            
        Returns:
            List of audio chunks
        """
        chunk_samples = int(chunk_duration * self.target_sample_rate)
        overlap_samples = int(chunk_samples * overlap)
        step_samples = chunk_samples - overlap_samples
        
        chunks = []
        start = 0
        
        while start < len(audio_data):
            end = min(start + chunk_samples, len(audio_data))
            chunk = audio_data[start:end]
            
            # Pad last chunk if needed
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            
            chunks.append(chunk)
            
            if end >= len(audio_data):
                break
                
            start += step_samples
        
        return chunks
    
    def get_audio_info(self, audio_path: Union[str, Path]) -> dict:
        """
        Get audio file information.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio information
        """
        try:
            # Use librosa for detailed info
            audio_data, sample_rate = librosa.load(str(audio_path), sr=None)
            
            duration = len(audio_data) / sample_rate
            
            # Get file size
            file_size = Path(audio_path).stat().st_size
            
            info = {
                'path': str(audio_path),
                'duration': duration,
                'sample_rate': sample_rate,
                'channels': 1 if audio_data.ndim == 1 else audio_data.shape[0],
                'samples': len(audio_data),
                'file_size': file_size,
                'format': Path(audio_path).suffix.lower(),
                'bit_depth': 'float32',  # librosa loads as float32
                'rms_level': float(np.sqrt(np.mean(audio_data ** 2))),
                'max_level': float(np.max(np.abs(audio_data)))
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get audio info for {audio_path}: {str(e)}")
            raise RuntimeError(f"Audio info extraction failed: {str(e)}")


class AudioValidator:
    """Validates audio files and data."""
    
    def __init__(self, processor: AudioProcessor):
        """
        Initialize audio validator.
        
        Args:
            processor: AudioProcessor instance
        """
        self.processor = processor
        self.logger = logging.getLogger(__name__)
    
    def validate_audio_file(self, audio_path: Union[str, Path]) -> dict:
        """
        Validate audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        try:
            # Check if file exists
            audio_path = Path(audio_path)
            if not audio_path.exists():
                validation_result['errors'].append(f"File does not exist: {audio_path}")
                return validation_result
            
            # Check file format
            if audio_path.suffix.lower() not in self.processor.supported_formats:
                validation_result['errors'].append(
                    f"Unsupported format: {audio_path.suffix}"
                )
                return validation_result
            
            # Get audio info
            info = self.processor.get_audio_info(audio_path)
            validation_result['info'] = info
            
            # Check duration
            if info['duration'] > self.processor.max_duration:
                validation_result['warnings'].append(
                    f"Duration ({info['duration']:.1f}s) exceeds maximum "
                    f"({self.processor.max_duration}s)"
                )
            
            # Check sample rate
            if info['sample_rate'] < 8000:
                validation_result['warnings'].append(
                    f"Low sample rate ({info['sample_rate']} Hz) may affect quality"
                )
            
            # Check audio level
            if info['max_level'] < 0.01:
                validation_result['warnings'].append("Audio level is very low")
            elif info['max_level'] > 0.99:
                validation_result['warnings'].append("Audio may be clipped")
            
            # If we get here, file is valid
            validation_result['valid'] = True
            
        except Exception as e:
            validation_result['errors'].append(str(e))
        
        return validation_result
    
    def validate_batch(self, audio_files: List[Union[str, Path]]) -> dict:
        """
        Validate multiple audio files.
        
        Args:
            audio_files: List of audio file paths
            
        Returns:
            Dictionary with batch validation results
        """
        results = {}
        valid_count = 0
        
        for audio_file in audio_files:
            result = self.validate_audio_file(audio_file)
            results[str(audio_file)] = result
            
            if result['valid']:
                valid_count += 1
        
        return {
            'total_files': len(audio_files),
            'valid_files': valid_count,
            'invalid_files': len(audio_files) - valid_count,
            'results': results
        }