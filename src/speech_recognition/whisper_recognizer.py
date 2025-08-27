"""
Speech Recognition Module using OpenAI Whisper

This module provides speech-to-text functionality with support for multiple languages
and automatic language detection.
"""

import os
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path

import whisper
import torch
import numpy as np
from whisper.utils import format_timestamp

from ..config import WHISPER_MODEL_SIZE, WHISPER_DEVICE
from ..audio_processing.processor import AudioProcessor


class SpeechRecognizer:
    """Speech recognition using OpenAI Whisper model."""
    
    def __init__(
        self, 
        model_size: str = WHISPER_MODEL_SIZE,
        device: str = WHISPER_DEVICE,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the speech recognizer.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run the model on (auto, cpu, cuda)
            cache_dir: Directory to cache downloaded models
        """
        self.model_size = model_size
        self.device = self._setup_device(device)
        self.cache_dir = cache_dir
        self.model = None
        self.audio_processor = AudioProcessor()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing SpeechRecognizer with model={model_size}, device={self.device}")
    
    def _setup_device(self, device: str) -> str:
        """Setup and validate device configuration."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        return device
    
    def load_model(self) -> None:
        """Load the Whisper model."""
        try:
            self.logger.info(f"Loading Whisper model: {self.model_size}")
            
            # Set cache directory if specified
            if self.cache_dir:
                os.environ['WHISPER_CACHE_DIR'] = self.cache_dir
            
            self.model = whisper.load_model(
                self.model_size, 
                device=self.device
            )
            
            self.logger.info("Whisper model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def transcribe(
        self, 
        audio_path: Union[str, Path], 
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            language: Source language code (optional, auto-detected if None)
            task: Task type ('transcribe' or 'translate')
            **kwargs: Additional arguments for whisper.transcribe()
        
        Returns:
            Dictionary containing transcription results
        """
        if self.model is None:
            self.load_model()
        
        try:
            # Preprocess audio
            audio_path = Path(audio_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            self.logger.info(f"Transcribing audio: {audio_path}")
            
            # Load and preprocess audio
            audio_data = self.audio_processor.load_audio(str(audio_path))
            
            # Prepare transcription options
            options = {
                "language": language,
                "task": task,
                "fp16": self.device == "cuda",
                **kwargs
            }
            
            # Remove None values
            options = {k: v for k, v in options.items() if v is not None}
            
            # Transcribe
            result = self.model.transcribe(audio_data, **options)
            
            # Process results
            processed_result = self._process_result(result, audio_path)
            
            self.logger.info(f"Transcription completed. Detected language: {processed_result['language']}")
            
            return processed_result
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {str(e)}")
            raise RuntimeError(f"Transcription failed: {str(e)}")
    
    def _process_result(self, result: Dict[str, Any], audio_path: Path) -> Dict[str, Any]:
        """Process and format transcription results."""
        
        # Extract segments with timestamps
        segments = []
        for segment in result.get("segments", []):
            segments.append({
                "id": segment["id"],
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
                "confidence": segment.get("avg_logprob", 0.0)
            })
        
        # Calculate confidence score
        confidence = self._calculate_confidence(result.get("segments", []))
        
        processed_result = {
            "text": result["text"].strip(),
            "language": result["language"],
            "segments": segments,
            "confidence": confidence,
            "audio_path": str(audio_path),
            "model_size": self.model_size,
            "processing_info": {
                "device": self.device,
                "num_segments": len(segments),
                "total_duration": segments[-1]["end"] if segments else 0.0
            }
        }
        
        return processed_result
    
    def _calculate_confidence(self, segments: list) -> float:
        """Calculate overall confidence score from segments."""
        if not segments:
            return 0.0
        
        total_confidence = sum(
            segment.get("avg_logprob", 0.0) 
            for segment in segments
        )
        
        # Convert log probabilities to confidence (0-1 scale)
        avg_logprob = total_confidence / len(segments)
        confidence = max(0.0, min(1.0, (avg_logprob + 1.0)))  # Normalize roughly
        
        return confidence
    
    def detect_language(self, audio_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Detect the language of the audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with language detection results
        """
        if self.model is None:
            self.load_model()
        
        try:
            audio_path = Path(audio_path)
            self.logger.info(f"Detecting language for: {audio_path}")
            
            # Load audio
            audio_data = self.audio_processor.load_audio(str(audio_path))
            
            # Detect language using Whisper's built-in detection
            # Use only first 30 seconds for faster detection
            audio_segment = audio_data[:30 * 16000]  # 30 seconds at 16kHz
            
            mel = whisper.log_mel_spectrogram(audio_segment).to(self.model.device)
            _, probs = self.model.detect_language(mel)
            
            # Get top 3 language predictions
            top_languages = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            
            result = {
                "detected_language": top_languages[0][0],
                "confidence": top_languages[0][1],
                "top_languages": [
                    {"language": lang, "confidence": conf}
                    for lang, conf in top_languages
                ],
                "audio_path": str(audio_path)
            }
            
            self.logger.info(f"Detected language: {result['detected_language']} "
                           f"(confidence: {result['confidence']:.3f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Language detection failed: {str(e)}")
            raise RuntimeError(f"Language detection failed: {str(e)}")
    
    def transcribe_with_timestamps(
        self, 
        audio_path: Union[str, Path], 
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio with detailed timestamp information.
        
        Args:
            audio_path: Path to audio file
            language: Source language code (optional)
            
        Returns:
            Dictionary with transcription and timestamp data
        """
        result = self.transcribe(
            audio_path, 
            language=language,
            word_timestamps=True,
            verbose=True
        )
        
        # Add formatted timestamps
        for segment in result["segments"]:
            segment["start_time"] = format_timestamp(segment["start"])
            segment["end_time"] = format_timestamp(segment["end"])
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "model_loaded": self.model is not None,
            "cache_dir": self.cache_dir,
            "cuda_available": torch.cuda.is_available()
        }


class BatchSpeechRecognizer:
    """Batch processing for multiple audio files."""
    
    def __init__(self, recognizer: SpeechRecognizer):
        """
        Initialize batch processor.
        
        Args:
            recognizer: SpeechRecognizer instance
        """
        self.recognizer = recognizer
        self.logger = logging.getLogger(__name__)
    
    def transcribe_batch(
        self, 
        audio_files: list, 
        language: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe multiple audio files.
        
        Args:
            audio_files: List of audio file paths
            language: Source language (optional)
            output_dir: Directory to save results (optional)
            
        Returns:
            Dictionary with batch processing results
        """
        results = {}
        failed_files = []
        
        self.logger.info(f"Starting batch transcription of {len(audio_files)} files")
        
        for i, audio_file in enumerate(audio_files, 1):
            try:
                self.logger.info(f"Processing file {i}/{len(audio_files)}: {audio_file}")
                
                result = self.recognizer.transcribe(audio_file, language=language)
                results[audio_file] = result
                
                # Save individual result if output directory specified
                if output_dir:
                    self._save_result(result, audio_file, output_dir)
                    
            except Exception as e:
                self.logger.error(f"Failed to process {audio_file}: {str(e)}")
                failed_files.append({"file": audio_file, "error": str(e)})
        
        batch_result = {
            "total_files": len(audio_files),
            "successful": len(results),
            "failed": len(failed_files),
            "results": results,
            "failed_files": failed_files
        }
        
        self.logger.info(f"Batch processing completed. "
                        f"Success: {batch_result['successful']}, "
                        f"Failed: {batch_result['failed']}")
        
        return batch_result
    
    def _save_result(self, result: Dict[str, Any], audio_file: str, output_dir: str) -> None:
        """Save individual transcription result to file."""
        import json
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create output filename
        audio_name = Path(audio_file).stem
        result_file = output_path / f"{audio_name}_transcription.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        self.logger.debug(f"Saved result to: {result_file}")


# Utility functions
def create_speech_recognizer(
    model_size: str = WHISPER_MODEL_SIZE,
    device: str = WHISPER_DEVICE
) -> SpeechRecognizer:
    """Create and initialize a speech recognizer."""
    recognizer = SpeechRecognizer(model_size=model_size, device=device)
    recognizer.load_model()
    return recognizer


def quick_transcribe(audio_path: str, language: Optional[str] = None) -> str:
    """Quick transcription function for simple use cases."""
    recognizer = create_speech_recognizer()
    result = recognizer.transcribe(audio_path, language=language)
    return result["text"]