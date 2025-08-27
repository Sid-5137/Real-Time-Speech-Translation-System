"""
Translation Module

This module provides text translation capabilities using multiple backends
including Google Translate API and local transformer models.
"""

import logging
import time
from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod

from googletrans import Translator as GoogleTranslator, LANGUAGES
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from ..config import DEFAULT_TRANSLATION_SERVICE, SUPPORTED_LANGUAGES


class TranslationEngine(ABC):
    """Abstract base class for translation engines."""
    
    @abstractmethod
    def translate(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Translate text from source language to target language."""
        pass
    
    @abstractmethod
    def detect_language(self, text: str) -> Dict[str, Any]:
        """Detect the language of input text."""
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported language codes and names."""
        pass


class GoogleTranslateEngine(TranslationEngine):
    """Google Translate API implementation."""
    
    def __init__(self, timeout: int = 10, retries: int = 3):
        """
        Initialize Google Translate engine.
        
        Args:
            timeout: Request timeout in seconds
            retries: Number of retry attempts
        """
        self.translator = GoogleTranslator()
        self.timeout = timeout
        self.retries = retries
        self.logger = logging.getLogger(__name__)
        
    def translate(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """
        Translate text using Google Translate.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Dictionary with translation results
        """
        if not text.strip():
            return {
                'text': text,
                'translated_text': text,
                'source_language': source_lang,
                'target_language': target_lang,
                'confidence': 1.0,
                'engine': 'google'
            }
        
        # Validate language codes
        self._validate_language_codes(source_lang, target_lang)
        
        for attempt in range(self.retries):
            try:
                self.logger.debug(f"Translating text (attempt {attempt + 1}): "
                                f"{source_lang} -> {target_lang}")
                
                # Perform translation
                result = self.translator.translate(
                    text, 
                    src=source_lang, 
                    dest=target_lang
                )
                
                # Extract results
                translation_result = {
                    'text': text,
                    'translated_text': result.text,
                    'source_language': result.src,
                    'target_language': target_lang,
                    'confidence': getattr(result, 'confidence', 0.95),
                    'engine': 'google',
                    'extra_data': result.extra_data if hasattr(result, 'extra_data') else {}
                }
                
                self.logger.debug(f"Translation successful: '{text}' -> '{result.text}'")
                return translation_result
                
            except Exception as e:
                self.logger.warning(f"Translation attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.retries - 1:
                    raise RuntimeError(f"Translation failed after {self.retries} attempts: {str(e)}")
                time.sleep(1)  # Wait before retry
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect language using Google Translate.
        
        Args:
            text: Text for language detection
            
        Returns:
            Dictionary with detection results
        """
        if not text.strip():
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'engine': 'google'
            }
        
        try:
            detection = self.translator.detect(text)
            
            return {
                'language': detection.lang,
                'confidence': detection.confidence,
                'engine': 'google',
                'text': text
            }
            
        except Exception as e:
            self.logger.error(f"Language detection failed: {str(e)}")
            raise RuntimeError(f"Language detection failed: {str(e)}")
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages from Google Translate."""
        return LANGUAGES
    
    def _validate_language_codes(self, source_lang: str, target_lang: str) -> None:
        """Validate language codes."""
        supported_languages = self.get_supported_languages()
        
        if source_lang not in supported_languages and source_lang != 'auto':
            raise ValueError(f"Unsupported source language: {source_lang}")
        
        if target_lang not in supported_languages:
            raise ValueError(f"Unsupported target language: {target_lang}")


class LocalTranslationEngine(TranslationEngine):
    """Local transformer model implementation."""
    
    def __init__(self, model_name: Optional[str] = None, device: str = "auto"):
        """
        Initialize local translation engine.
        
        Args:
            model_name: Hugging Face model name (uses default if None)
            device: Device to run model on (auto, cpu, cuda)
        """
        self.device = self._setup_device(device)
        self.model_name = model_name or "Helsinki-NLP/opus-mt-en-mul"
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        self.logger = logging.getLogger(__name__)
        
        # Language mapping for Helsinki models
        self.language_mapping = {
            'en': 'eng',
            'es': 'spa', 
            'fr': 'fra',
            'de': 'deu',
            'it': 'ita',
            'pt': 'por',
            'ru': 'rus'
        }
        
    def _setup_device(self, device: str) -> str:
        """Setup device configuration."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_model(self) -> None:
        """Load the translation model."""
        try:
            self.logger.info(f"Loading translation model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # Create pipeline for easier use
            self.pipeline = pipeline(
                "translation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            self.logger.info("Translation model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load translation model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """
        Translate text using local model.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Dictionary with translation results
        """
        if self.pipeline is None:
            self.load_model()
        
        if not text.strip():
            return {
                'text': text,
                'translated_text': text,
                'source_language': source_lang,
                'target_language': target_lang,
                'confidence': 1.0,
                'engine': 'local'
            }
        
        try:
            # Prepare input for Helsinki models (may need language prefixes)
            input_text = self._prepare_input(text, target_lang)
            
            # Perform translation
            results = self.pipeline(input_text, max_length=512)
            
            if isinstance(results, list) and len(results) > 0:
                translated_text = results[0]['translation_text']
            else:
                translated_text = results['translation_text']
            
            # Clean up output
            translated_text = self._clean_output(translated_text)
            
            return {
                'text': text,
                'translated_text': translated_text,
                'source_language': source_lang,
                'target_language': target_lang,
                'confidence': 0.85,  # Placeholder confidence for local models
                'engine': 'local',
                'model_name': self.model_name
            }
            
        except Exception as e:
            self.logger.error(f"Local translation failed: {str(e)}")
            raise RuntimeError(f"Local translation failed: {str(e)}")
    
    def _prepare_input(self, text: str, target_lang: str) -> str:
        """Prepare input text for translation (add language prefixes if needed)."""
        # For Helsinki models, may need to add target language prefix
        if "Helsinki-NLP" in self.model_name:
            # Some Helsinki models use language codes as prefixes
            mapped_lang = self.language_mapping.get(target_lang, target_lang)
            return f">>{mapped_lang}<< {text}"
        return text
    
    def _clean_output(self, text: str) -> str:
        """Clean translation output."""
        # Remove any language prefixes that might be in output
        for lang_code in self.language_mapping.values():
            prefix = f">>{lang_code}<< "
            if text.startswith(prefix):
                text = text[len(prefix):]
        return text.strip()
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect language (placeholder - local models don't typically do detection).
        
        Args:
            text: Text for language detection
            
        Returns:
            Dictionary with detection results
        """
        # Most local translation models don't include language detection
        # This is a placeholder that could be enhanced with a separate detection model
        
        self.logger.warning("Language detection not implemented for local models")
        return {
            'language': 'unknown',
            'confidence': 0.0,
            'engine': 'local',
            'note': 'Language detection not available with local models'
        }
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages for local model."""
        # Return basic supported languages - could be enhanced by parsing model config
        return {code: name for code, name in SUPPORTED_LANGUAGES.items() 
                if code in self.language_mapping}


class TranslationService:
    """Main translation service that manages multiple engines."""
    
    def __init__(
        self, 
        primary_engine: str = DEFAULT_TRANSLATION_SERVICE,
        fallback_engine: Optional[str] = None
    ):
        """
        Initialize translation service.
        
        Args:
            primary_engine: Primary translation engine ('google' or 'local')
            fallback_engine: Fallback engine if primary fails
        """
        self.primary_engine_name = primary_engine
        self.fallback_engine_name = fallback_engine
        
        self.engines = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize engines
        self._initialize_engines()
    
    def _initialize_engines(self) -> None:
        """Initialize translation engines."""
        try:
            # Initialize Google Translate engine
            self.engines['google'] = GoogleTranslateEngine()
            self.logger.info("Google Translate engine initialized")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize Google Translate: {str(e)}")
        
        try:
            # Initialize local engine
            self.engines['local'] = LocalTranslationEngine()
            self.logger.info("Local translation engine initialized")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize local engine: {str(e)}")
    
    def translate(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str,
        engine: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Translate text with automatic fallback.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            engine: Specific engine to use (optional)
            
        Returns:
            Dictionary with translation results
        """
        # Determine which engine to use
        engine_name = engine or self.primary_engine_name
        
        # Try primary engine
        try:
            if engine_name in self.engines:
                return self.engines[engine_name].translate(text, source_lang, target_lang)
            else:
                raise ValueError(f"Engine '{engine_name}' not available")
                
        except Exception as e:
            self.logger.warning(f"Primary engine '{engine_name}' failed: {str(e)}")
            
            # Try fallback engine if available
            if (self.fallback_engine_name and 
                self.fallback_engine_name in self.engines and 
                self.fallback_engine_name != engine_name):
                
                try:
                    self.logger.info(f"Trying fallback engine: {self.fallback_engine_name}")
                    return self.engines[self.fallback_engine_name].translate(
                        text, source_lang, target_lang
                    )
                except Exception as fallback_error:
                    self.logger.error(f"Fallback engine also failed: {str(fallback_error)}")
            
            # If all engines fail, raise the original error
            raise RuntimeError(f"Translation failed: {str(e)}")
    
    def detect_language(self, text: str, engine: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect text language.
        
        Args:
            text: Text for language detection
            engine: Specific engine to use (optional)
            
        Returns:
            Dictionary with detection results
        """
        engine_name = engine or self.primary_engine_name
        
        if engine_name in self.engines:
            return self.engines[engine_name].detect_language(text)
        else:
            raise ValueError(f"Engine '{engine_name}' not available")
    
    def batch_translate(
        self, 
        texts: List[str], 
        source_lang: str, 
        target_lang: str,
        engine: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Translate multiple texts.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            engine: Specific engine to use (optional)
            
        Returns:
            List of translation results
        """
        results = []
        
        for i, text in enumerate(texts):
            try:
                self.logger.debug(f"Translating text {i+1}/{len(texts)}")
                result = self.translate(text, source_lang, target_lang, engine)
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to translate text {i+1}: {str(e)}")
                # Add error result
                results.append({
                    'text': text,
                    'translated_text': text,  # Fallback to original
                    'source_language': source_lang,
                    'target_language': target_lang,
                    'confidence': 0.0,
                    'engine': 'error',
                    'error': str(e)
                })
        
        return results
    
    def get_available_engines(self) -> List[str]:
        """Get list of available engines."""
        return list(self.engines.keys())
    
    def get_supported_languages(self, engine: Optional[str] = None) -> Dict[str, str]:
        """
        Get supported languages.
        
        Args:
            engine: Specific engine (uses primary if None)
            
        Returns:
            Dictionary of language codes and names
        """
        engine_name = engine or self.primary_engine_name
        
        if engine_name in self.engines:
            return self.engines[engine_name].get_supported_languages()
        else:
            return SUPPORTED_LANGUAGES


# Utility functions
def create_translation_service(
    primary_engine: str = DEFAULT_TRANSLATION_SERVICE,
    fallback_engine: str = "google"
) -> TranslationService:
    """Create and initialize translation service."""
    return TranslationService(primary_engine, fallback_engine)


def quick_translate(
    text: str, 
    source_lang: str, 
    target_lang: str, 
    engine: str = DEFAULT_TRANSLATION_SERVICE
) -> str:
    """Quick translation function for simple use cases."""
    service = create_translation_service(primary_engine=engine)
    result = service.translate(text, source_lang, target_lang)
    return result['translated_text']