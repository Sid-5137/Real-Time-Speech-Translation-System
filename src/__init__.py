"""
Speech Translation System with Voice Cloning

A comprehensive system for translating speech while preserving voice characteristics.
"""

__version__ = "1.0.0"
__author__ = "Speech Translation Team"

from .pipeline.main_pipeline import SpeechTranslator

__all__ = ["SpeechTranslator"]