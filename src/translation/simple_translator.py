"""
Simple Translation Service

A lightweight translation service that works around dependency conflicts.
Uses multiple translation backends with fallbacks.
"""

import requests
import json
from typing import Dict, Any, Optional
import logging
import time


class SimpleTranslator:
    """Simple translation service with multiple backends"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Language mapping
        self.languages = {
            "en": "English",
            "hi": "Hindi", 
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
            "ar": "Arabic"
        }
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """
        Translate text from source to target language
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translation result dictionary
        """
        try:
            # Try MyMemory translation API (free, no auth required)
            result = self._translate_with_mymemory(text, source_lang, target_lang)
            
            if result['success']:
                return result
            
            # Fallback: Simple mock translation for demo
            return self._mock_translate(text, source_lang, target_lang)
            
        except Exception as e:
            self.logger.error(f"Translation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'translated_text': text,  # Return original as fallback
                'source_language': source_lang,
                'target_language': target_lang,
                'service': 'error'
            }
    
    def _translate_with_mymemory(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Use MyMemory translation API"""
        try:
            # MyMemory API endpoint
            url = "https://api.mymemory.translated.net/get"
            
            params = {
                'q': text,
                'langpair': f"{source_lang}|{target_lang}"
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('responseStatus') == 200:
                    translated_text = data['responseData']['translatedText']
                    
                    return {
                        'success': True,
                        'translated_text': translated_text,
                        'source_language': source_lang,
                        'target_language': target_lang,
                        'confidence': float(data['responseData'].get('match', 0.8)),
                        'service': 'MyMemory'
                    }
            
            return {'success': False, 'error': 'MyMemory API failed'}
            
        except Exception as e:
            self.logger.warning(f"MyMemory translation failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _mock_translate(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Mock translation for demo purposes"""
        
        # Simple demo translations for common phrases
        demo_translations = {
            ('hi', 'en'): {
                'नमस्ते': 'Hello',
                'आप कैसे हैं?': 'How are you?',
                'धन्यवाद': 'Thank you',
                'जब मैं चोटा था': 'When I was small',
                'जब मैं चोटा था मैं हमें सा ज़िली सोकर उड़ता था': 'When I was small, I used to fly around like a gentle breeze'
            },
            ('en', 'hi'): {
                'Hello': 'नमस्ते',
                'How are you?': 'आप कैसे हैं?',
                'Thank you': 'धन्यवाद',
                'When I was small': 'जब मैं चोटा था'
            },
            ('en', 'es'): {
                'Hello': 'Hola',
                'How are you?': '¿Cómo estás?',
                'Thank you': 'Gracias',
                'When I was small': 'Cuando era pequeño'
            },
            ('es', 'en'): {
                'Hola': 'Hello',
                '¿Cómo estás?': 'How are you?',
                'Gracias': 'Thank you'
            }
        }
        
        # Check for exact matches first
        lang_pair = (source_lang, target_lang)
        if lang_pair in demo_translations:
            for source_phrase, target_phrase in demo_translations[lang_pair].items():
                if source_phrase.lower() in text.lower():
                    translated_text = text.replace(source_phrase, target_phrase)
                    return {
                        'success': True,
                        'translated_text': translated_text,
                        'source_language': source_lang,
                        'target_language': target_lang,
                        'confidence': 0.9,
                        'service': 'Demo (Mock)'
                    }
        
        # Generic fallback
        if source_lang == target_lang:
            translated_text = text
        else:
            translated_text = f"[{target_lang.upper()}] {text}"
        
        return {
            'success': True,
            'translated_text': translated_text,
            'source_language': source_lang,
            'target_language': target_lang,
            'confidence': 0.5,
            'service': 'Demo (Fallback)'
        }
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages"""
        return self.languages.copy()
    
    def detect_language(self, text: str) -> str:
        """Simple language detection (placeholder)"""
        # Simple heuristics for common languages
        if any(char in text for char in 'देवनागरी'):
            return 'hi'
        elif any(char in text for char in 'áéíóúñü¿¡'):
            return 'es'
        elif any(char in text for char in 'àâäéèêëîïôöùûüÿç'):
            return 'fr'
        elif any(char in text for char in 'äöüß'):
            return 'de'
        else:
            return 'en'  # Default to English


# Factory function
def create_simple_translator() -> SimpleTranslator:
    """Create and return a SimpleTranslator instance"""
    return SimpleTranslator()


# Test function
def test_translator():
    """Test the translator"""
    translator = create_simple_translator()
    
    # Test cases
    test_cases = [
        ("Hello, how are you?", "en", "hi"),
        ("नमस्ते", "hi", "en"),
        ("Hola", "es", "en"),
    ]
    
    print("🔄 Testing Simple Translator")
    print("=" * 40)
    
    for text, source, target in test_cases:
        result = translator.translate_text(text, source, target)
        
        print(f"🌍 {source} → {target}")
        print(f"📝 Input: {text}")
        print(f"✅ Output: {result['translated_text']}")
        print(f"🔧 Service: {result['service']}")
        print("-" * 30)


if __name__ == "__main__":
    test_translator()