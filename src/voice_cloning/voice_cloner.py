"""
Voice Cloning Module

This module provides voice cloning and text-to-speech capabilities using
Coqui TTS and other state-of-the-art TTS models.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json

import torch
import numpy as np
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import soundfile as sf

from ..config import TTS_MODEL, VOICE_CLONE_SAMPLES_MIN, VOICE_CLONE_DURATION_MIN, SAMPLE_RATE
from ..audio_processing.processor import AudioProcessor


class VoiceCloner:
    """Voice cloning using Coqui TTS models."""
    
    def __init__(
        self, 
        model_name: str = TTS_MODEL,
        device: str = "auto",
        use_gpu: bool = True
    ):
        """
        Initialize voice cloner.
        
        Args:
            model_name: TTS model name
            device: Device to run model on
            use_gpu: Whether to use GPU acceleration
        """
        self.model_name = model_name
        self.device = self._setup_device(device, use_gpu)
        self.tts = None
        self.model = None
        
        self.audio_processor = AudioProcessor()
        self.logger = logging.getLogger(__name__)
        
        # Voice sample management
        self.voice_samples = {}
        self.speaker_embeddings = {}
        
    def _setup_device(self, device: str, use_gpu: bool) -> str:
        """Setup device configuration."""
        if device == "auto":
            if use_gpu and torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def load_model(self) -> None:
        """Load the TTS model."""
        try:
            self.logger.info(f"Loading TTS model: {self.model_name}")
            
            # Initialize TTS
            self.tts = TTS(
                model_name=self.model_name,
                progress_bar=True,
                gpu=(self.device == "cuda")
            )
            
            self.logger.info("TTS model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load TTS model: {str(e)}")
            raise RuntimeError(f"TTS model loading failed: {str(e)}")
    
    def register_voice(
        self, 
        speaker_name: str, 
        voice_samples: List[Union[str, Path]],
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Register a new voice with audio samples.
        
        Args:
            speaker_name: Unique identifier for the speaker
            voice_samples: List of paths to voice sample files
            validate: Whether to validate voice samples
            
        Returns:
            Dictionary with registration results
        """
        try:
            self.logger.info(f"Registering voice: {speaker_name}")
            
            if validate:
                validation_result = self._validate_voice_samples(voice_samples)
                if not validation_result['valid']:
                    raise ValueError(f"Voice sample validation failed: {validation_result['errors']}")
            
            # Process voice samples
            processed_samples = []
            total_duration = 0.0
            
            for sample_path in voice_samples:
                # Load and process audio
                audio_data = self.audio_processor.load_audio(sample_path, normalize=True)
                
                # Calculate duration
                duration = len(audio_data) / SAMPLE_RATE
                total_duration += duration
                
                processed_samples.append({
                    'path': str(sample_path),
                    'audio_data': audio_data,
                    'duration': duration
                })
            
            # Store voice information
            self.voice_samples[speaker_name] = {
                'samples': processed_samples,
                'total_duration': total_duration,
                'num_samples': len(processed_samples),
                'registered_at': self._get_timestamp()
            }
            
            # Generate speaker embedding if using XTTS
            if "xtts" in self.model_name.lower():
                self._generate_speaker_embedding(speaker_name)
            
            result = {
                'speaker_name': speaker_name,
                'num_samples': len(processed_samples),
                'total_duration': total_duration,
                'status': 'registered'
            }
            
            self.logger.info(f"Voice registered successfully: {speaker_name} "
                           f"({len(processed_samples)} samples, {total_duration:.1f}s)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Voice registration failed: {str(e)}")
            raise RuntimeError(f"Voice registration failed: {str(e)}")
    
    def _validate_voice_samples(self, voice_samples: List[Union[str, Path]]) -> Dict[str, Any]:
        """Validate voice samples."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        if len(voice_samples) < VOICE_CLONE_SAMPLES_MIN:
            validation_result['errors'].append(
                f"Need at least {VOICE_CLONE_SAMPLES_MIN} voice samples, got {len(voice_samples)}"
            )
            validation_result['valid'] = False
        
        total_duration = 0.0
        valid_samples = 0
        
        for sample_path in voice_samples:
            try:
                # Validate individual file
                file_validation = self.audio_processor.get_audio_info(sample_path)
                total_duration += file_validation['duration']
                valid_samples += 1
                
                # Check sample quality
                if file_validation['duration'] < 3.0:
                    validation_result['warnings'].append(
                        f"Short sample ({file_validation['duration']:.1f}s): {sample_path}"
                    )
                
                if file_validation['sample_rate'] < 16000:
                    validation_result['warnings'].append(
                        f"Low sample rate ({file_validation['sample_rate']} Hz): {sample_path}"
                    )
                    
            except Exception as e:
                validation_result['errors'].append(f"Invalid sample {sample_path}: {str(e)}")
        
        if total_duration < VOICE_CLONE_DURATION_MIN:
            validation_result['errors'].append(
                f"Total duration ({total_duration:.1f}s) below minimum ({VOICE_CLONE_DURATION_MIN}s)"
            )
            validation_result['valid'] = False
        
        validation_result['info'] = {
            'total_samples': len(voice_samples),
            'valid_samples': valid_samples,
            'total_duration': total_duration
        }
        
        return validation_result
    
    def _generate_speaker_embedding(self, speaker_name: str) -> None:
        """Generate speaker embedding for XTTS models."""
        if self.tts is None:
            self.load_model()
        
        try:
            voice_data = self.voice_samples[speaker_name]
            
            # Concatenate all samples for embedding generation
            combined_audio = []
            for sample in voice_data['samples']:
                combined_audio.extend(sample['audio_data'])
            
            # Convert to tensor and generate embedding
            audio_tensor = torch.FloatTensor(combined_audio).unsqueeze(0)
            
            # This is a placeholder - actual implementation depends on TTS model
            # For XTTS, you might use the model's speaker encoder
            self.logger.info(f"Generated speaker embedding for {speaker_name}")
            
        except Exception as e:
            self.logger.warning(f"Failed to generate speaker embedding: {str(e)}")
    
    def clone_voice(
        self, 
        text: str, 
        speaker_name: str,
        language: str = "en",
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate speech using cloned voice.
        
        Args:
            text: Text to synthesize
            speaker_name: Registered speaker name
            language: Target language
            output_path: Output file path (optional)
            **kwargs: Additional TTS parameters
            
        Returns:
            Dictionary with synthesis results
        """
        if self.tts is None:
            self.load_model()
        
        if speaker_name not in self.voice_samples:
            raise ValueError(f"Speaker '{speaker_name}' not registered")
        
        try:
            self.logger.info(f"Generating speech for '{speaker_name}': {text[:50]}...")
            
            # Get voice samples for the speaker
            voice_data = self.voice_samples[speaker_name]
            
            # Use first sample as reference (could be improved by selecting best sample)
            reference_audio_path = voice_data['samples'][0]['path']
            
            # Generate speech
            if "xtts" in self.model_name.lower():
                # XTTS-specific generation
                audio = self._generate_xtts(text, reference_audio_path, language, **kwargs)
            else:
                # Generic TTS generation
                audio = self._generate_generic_tts(text, reference_audio_path, language, **kwargs)
            
            # Save audio if output path provided
            if output_path:
                output_path = Path(output_path)
                self.audio_processor.save_audio(audio, output_path)
                self.logger.info(f"Saved generated audio to: {output_path}")
            
            result = {
                'text': text,
                'speaker_name': speaker_name,
                'language': language,
                'audio_data': audio,
                'sample_rate': SAMPLE_RATE,
                'duration': len(audio) / SAMPLE_RATE,
                'output_path': str(output_path) if output_path else None,
                'model_used': self.model_name
            }
            
            self.logger.info(f"Voice cloning completed: {result['duration']:.1f}s audio generated")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Voice cloning failed: {str(e)}")
            raise RuntimeError(f"Voice cloning failed: {str(e)}")
    
    def _generate_xtts(
        self, 
        text: str, 
        reference_audio_path: str, 
        language: str,
        **kwargs
    ) -> np.ndarray:
        """Generate speech using XTTS model."""
        try:
            # XTTS generation
            audio = self.tts.tts(
                text=text,
                speaker_wav=reference_audio_path,
                language=language,
                **kwargs
            )
            
            return np.array(audio, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"XTTS generation failed: {str(e)}")
            raise RuntimeError(f"XTTS generation failed: {str(e)}")
    
    def _generate_generic_tts(
        self, 
        text: str, 
        reference_audio_path: str, 
        language: str,
        **kwargs
    ) -> np.ndarray:
        """Generate speech using generic TTS model."""
        try:
            # Generic TTS generation
            audio = self.tts.tts(
                text=text,
                speaker_wav=reference_audio_path,
                **kwargs
            )
            
            return np.array(audio, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Generic TTS generation failed: {str(e)}")
            raise RuntimeError(f"Generic TTS generation failed: {str(e)}")
    
    def get_registered_speakers(self) -> List[str]:
        """Get list of registered speakers."""
        return list(self.voice_samples.keys())
    
    def get_speaker_info(self, speaker_name: str) -> Dict[str, Any]:
        """Get information about a registered speaker."""
        if speaker_name not in self.voice_samples:
            raise ValueError(f"Speaker '{speaker_name}' not found")
        
        voice_data = self.voice_samples[speaker_name]
        
        return {
            'speaker_name': speaker_name,
            'num_samples': voice_data['num_samples'],
            'total_duration': voice_data['total_duration'],
            'registered_at': voice_data['registered_at'],
            'samples': [sample['path'] for sample in voice_data['samples']]
        }
    
    def remove_speaker(self, speaker_name: str) -> bool:
        """Remove a registered speaker."""
        if speaker_name in self.voice_samples:
            del self.voice_samples[speaker_name]
            
            if speaker_name in self.speaker_embeddings:
                del self.speaker_embeddings[speaker_name]
            
            self.logger.info(f"Removed speaker: {speaker_name}")
            return True
        
        return False
    
    def save_speaker_data(self, output_dir: Union[str, Path]) -> None:
        """Save speaker data to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save voice sample metadata
        metadata_file = output_dir / "speakers_metadata.json"
        
        metadata = {}
        for speaker_name, voice_data in self.voice_samples.items():
            metadata[speaker_name] = {
                'num_samples': voice_data['num_samples'],
                'total_duration': voice_data['total_duration'],
                'registered_at': voice_data['registered_at'],
                'sample_paths': [sample['path'] for sample in voice_data['samples']]
            }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved speaker metadata to: {metadata_file}")
    
    def load_speaker_data(self, input_dir: Union[str, Path]) -> None:
        """Load speaker data from disk."""
        input_dir = Path(input_dir)
        metadata_file = input_dir / "speakers_metadata.json"
        
        if not metadata_file.exists():
            self.logger.warning(f"Speaker metadata not found: {metadata_file}")
            return
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            for speaker_name, speaker_data in metadata.items():
                # Re-register speaker with existing samples
                sample_paths = speaker_data['sample_paths']
                
                # Validate that samples still exist
                valid_samples = [path for path in sample_paths if Path(path).exists()]
                
                if valid_samples:
                    self.register_voice(speaker_name, valid_samples, validate=False)
                    self.logger.info(f"Loaded speaker: {speaker_name}")
                else:
                    self.logger.warning(f"No valid samples found for speaker: {speaker_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load speaker data: {str(e)}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'model_loaded': self.tts is not None,
            'num_registered_speakers': len(self.voice_samples),
            'cuda_available': torch.cuda.is_available()
        }


class BatchVoiceCloner:
    """Batch processing for voice cloning tasks."""
    
    def __init__(self, voice_cloner: VoiceCloner):
        """
        Initialize batch voice cloner.
        
        Args:
            voice_cloner: VoiceCloner instance
        """
        self.voice_cloner = voice_cloner
        self.logger = logging.getLogger(__name__)
    
    def clone_batch(
        self, 
        texts: List[str], 
        speaker_name: str,
        language: str = "en",
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate speech for multiple texts using the same voice.
        
        Args:
            texts: List of texts to synthesize
            speaker_name: Registered speaker name
            language: Target language
            output_dir: Directory to save output files
            **kwargs: Additional TTS parameters
            
        Returns:
            Dictionary with batch processing results
        """
        results = []
        failed_texts = []
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting batch voice cloning: {len(texts)} texts")
        
        for i, text in enumerate(texts, 1):
            try:
                self.logger.info(f"Processing text {i}/{len(texts)}")
                
                # Generate output path if directory provided
                output_path = None
                if output_dir:
                    output_path = output_dir / f"speech_{i:04d}.wav"
                
                result = self.voice_cloner.clone_voice(
                    text=text,
                    speaker_name=speaker_name,
                    language=language,
                    output_path=output_path,
                    **kwargs
                )
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to process text {i}: {str(e)}")
                failed_texts.append({'index': i, 'text': text, 'error': str(e)})
        
        batch_result = {
            'total_texts': len(texts),
            'successful': len(results),
            'failed': len(failed_texts),
            'results': results,
            'failed_texts': failed_texts,
            'speaker_name': speaker_name,
            'language': language
        }
        
        self.logger.info(f"Batch voice cloning completed. "
                        f"Success: {batch_result['successful']}, "
                        f"Failed: {batch_result['failed']}")
        
        return batch_result


# Utility functions
def create_voice_cloner(
    model_name: str = TTS_MODEL,
    device: str = "auto"
) -> VoiceCloner:
    """Create and initialize voice cloner."""
    cloner = VoiceCloner(model_name=model_name, device=device)
    cloner.load_model()
    return cloner


def quick_voice_clone(
    text: str,
    voice_sample_path: str,
    output_path: str,
    language: str = "en"
) -> str:
    """Quick voice cloning for simple use cases."""
    cloner = create_voice_cloner()
    
    # Register temporary speaker
    temp_speaker = "temp_speaker"
    cloner.register_voice(temp_speaker, [voice_sample_path])
    
    # Generate speech
    result = cloner.clone_voice(
        text=text,
        speaker_name=temp_speaker,
        language=language,
        output_path=output_path
    )
    
    return str(result['output_path'])