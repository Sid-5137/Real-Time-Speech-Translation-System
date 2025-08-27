"""
Example Usage Scripts for Speech Translation System

This file demonstrates how to use the speech translation system
in various scenarios.
"""

import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline.main_pipeline import create_speech_translator, quick_translate_audio
from src.voice_cloning.voice_cloner import create_voice_cloner
from src.speech_recognition.whisper_recognizer import create_speech_recognizer
from src.translation.translator import create_translation_service

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_translation():
    """
    Example 1: Basic audio translation with voice cloning
    """
    print("\\n" + "="*60)
    print("Example 1: Basic Audio Translation")
    print("="*60)
    
    try:
        # Initialize the speech translator
        translator = create_speech_translator()
        
        # Example file paths (you'll need to provide actual files)
        input_audio = "data/samples/input_speech.wav"
        voice_sample = "data/voice_samples/target_voice.wav" 
        output_path = "data/samples/translated_output.wav"
        
        # Check if files exist
        if not Path(input_audio).exists():
            print(f"⚠️  Input audio file not found: {input_audio}")
            print("Please provide an actual audio file to test with.")
            return
        
        if not Path(voice_sample).exists():
            print(f"⚠️  Voice sample file not found: {voice_sample}")
            print("Please provide a voice sample file to test with.")
            return
        
        print(f"🎙️  Input: {input_audio}")
        print(f"🎯 Voice Sample: {voice_sample}")
        print(f"🌍 Translation: auto → English")
        
        # Perform translation
        result = translator.translate_audio(
            input_audio=input_audio,
            target_lang="en",
            voice_sample=voice_sample,
            output_path=output_path,
            return_intermediate=True
        )
        
        if result['success']:
            print("\\n✅ Translation successful!")
            print(f"📝 Original: {result['original_text']}")
            print(f"📝 Translated: {result['translated_text']}")
            print(f"🎵 Output: {result['output_audio']}")
            print(f"⏱️  Processing time: {result['processing_time']:.2f}s")
        else:
            print(f"❌ Translation failed: {result['error']}")
    
    except Exception as e:
        print(f"💥 Error: {str(e)}")
        logger.exception("Example 1 failed")


def example_2_text_to_speech():
    """
    Example 2: Text translation with voice cloning
    """
    print("\\n" + "="*60)
    print("Example 2: Text to Speech Translation")
    print("="*60)
    
    try:
        # Initialize translator
        translator = create_speech_translator()
        
        # Example text and voice sample
        text = "Hello, how are you doing today?"
        voice_sample = "data/voice_samples/target_voice.wav"
        output_path = "data/samples/text_to_speech_output.wav"
        
        if not Path(voice_sample).exists():
            print(f"⚠️  Voice sample file not found: {voice_sample}")
            print("Please provide a voice sample file to test with.")
            return
        
        print(f"📝 Text: {text}")
        print(f"🎯 Voice Sample: {voice_sample}")
        print(f"🌍 Translation: English → Spanish")
        
        # Perform text translation and speech generation
        result = translator.translate_text_with_voice(
            text=text,
            source_lang="en",
            target_lang="es", 
            voice_sample=voice_sample,
            output_path=output_path
        )
        
        if result['success']:
            print("\\n✅ Text-to-speech translation successful!")
            print(f"📝 Original: {result['original_text']}")
            print(f"📝 Translated: {result['translated_text']}")
            print(f"🎵 Output: {result['output_audio']}")
        else:
            print(f"❌ Translation failed: {result['error']}")
    
    except Exception as e:
        print(f"💥 Error: {str(e)}")
        logger.exception("Example 2 failed")


def example_3_speaker_registration():
    """
    Example 3: Register and reuse speaker voices
    """
    print("\\n" + "="*60)
    print("Example 3: Speaker Voice Registration")
    print("="*60)
    
    try:
        # Initialize voice cloner
        cloner = create_voice_cloner()
        
        # Example voice samples for a speaker
        speaker_name = "john_doe"
        voice_samples = [
            "data/voice_samples/john_sample_1.wav",
            "data/voice_samples/john_sample_2.wav", 
            "data/voice_samples/john_sample_3.wav"
        ]
        
        # Check if samples exist
        existing_samples = [sample for sample in voice_samples if Path(sample).exists()]
        if not existing_samples:
            print("⚠️  No voice sample files found.")
            print("Please provide voice sample files in data/voice_samples/")
            return
        
        print(f"👤 Registering speaker: {speaker_name}")
        print(f"🎵 Voice samples: {len(existing_samples)} files")
        
        # Register the speaker
        result = cloner.register_voice(speaker_name, existing_samples)
        
        print(f"\\n✅ Speaker registered successfully!")
        print(f"👤 Name: {result['speaker_name']}")
        print(f"🎵 Samples: {result['num_samples']}")
        print(f"⏱️  Total duration: {result['total_duration']:.1f}s")
        
        # Now use the registered speaker for translation
        translator = create_speech_translator()
        
        text = "This is a test using my registered voice."
        output_path = "data/samples/registered_speaker_output.wav"
        
        result = translator.translate_text_with_voice(
            text=text,
            source_lang="en",
            target_lang="fr",
            speaker_name=speaker_name,  # Use registered speaker
            output_path=output_path
        )
        
        if result['success']:
            print(f"\\n🎵 Generated speech with registered voice!")
            print(f"📝 Translated text: {result['translated_text']}")
            print(f"🎵 Output: {result['output_audio']}")
    
    except Exception as e:
        print(f"💥 Error: {str(e)}")
        logger.exception("Example 3 failed")


def example_4_batch_processing():
    """
    Example 4: Batch processing multiple files
    """
    print("\\n" + "="*60)
    print("Example 4: Batch Processing")
    print("="*60)
    
    try:
        # Initialize translator
        translator = create_speech_translator()
        
        # Example batch of audio files
        audio_files = [
            "data/samples/audio_1.wav",
            "data/samples/audio_2.wav",
            "data/samples/audio_3.wav"
        ]
        
        voice_sample = "data/voice_samples/target_voice.wav"
        output_dir = "data/samples/batch_output"
        
        # Check if files exist
        existing_files = [f for f in audio_files if Path(f).exists()]
        if not existing_files:
            print("⚠️  No audio files found for batch processing.")
            print("Please provide audio files in data/samples/")
            return
        
        if not Path(voice_sample).exists():
            print(f"⚠️  Voice sample file not found: {voice_sample}")
            return
        
        print(f"📦 Processing {len(existing_files)} files")
        print(f"🎯 Voice sample: {voice_sample}")
        print(f"🌍 Target language: Spanish")
        print(f"💾 Output directory: {output_dir}")
        
        # Perform batch processing
        result = translator.batch_translate_audio(
            audio_files=existing_files,
            target_lang="es",
            voice_sample=voice_sample,
            output_dir=output_dir
        )
        
        print(f"\\n📊 Batch processing completed!")
        print(f"✅ Successful: {result['successful']}")
        print(f"❌ Failed: {result['failed']}")
        
        if result['failed_files']:
            print("\\n🚨 Failed files:")
            for failed in result['failed_files']:
                print(f"  - {failed['file']}: {failed['error']}")
    
    except Exception as e:
        print(f"💥 Error: {str(e)}")
        logger.exception("Example 4 failed")


def example_5_component_usage():
    """
    Example 5: Using individual components separately
    """
    print("\\n" + "="*60)
    print("Example 5: Individual Component Usage")
    print("="*60)
    
    try:
        print("🔧 Demonstrating individual component usage...")
        
        # 1. Speech Recognition only
        print("\\n1️⃣ Speech Recognition:")
        recognizer = create_speech_recognizer()
        
        input_audio = "data/samples/input_speech.wav"
        if Path(input_audio).exists():
            result = recognizer.transcribe(input_audio)
            print(f"📝 Transcribed: {result['text'][:100]}...")
            print(f"🌍 Language: {result['language']}")
        else:
            print("⚠️  No input audio file found")
        
        # 2. Translation only
        print("\\n2️⃣ Translation:")
        translation_service = create_translation_service()
        
        text = "Hello, how are you?"
        result = translation_service.translate(text, "en", "es")
        print(f"📝 Original: {result['text']}")
        print(f"📝 Translated: {result['translated_text']}")
        print(f"🔧 Engine: {result['engine']}")
        
        # 3. Voice Cloning only
        print("\\n3️⃣ Voice Cloning:")
        cloner = create_voice_cloner()
        
        voice_sample = "data/voice_samples/target_voice.wav"
        if Path(voice_sample).exists():
            # Register a temporary speaker
            cloner.register_voice("temp_speaker", [voice_sample])
            
            # Generate speech
            result = cloner.clone_voice(
                text="Hola, ¿cómo estás?",
                speaker_name="temp_speaker",
                language="es",
                output_path="data/samples/voice_clone_output.wav"
            )
            
            print(f"🎵 Generated: {result['duration']:.1f}s of audio")
            print(f"💾 Saved to: {result['output_path']}")
        else:
            print("⚠️  No voice sample file found")
    
    except Exception as e:
        print(f"💥 Error: {str(e)}")
        logger.exception("Example 5 failed")


def example_6_system_info():
    """
    Example 6: System information and configuration
    """
    print("\\n" + "="*60)
    print("Example 6: System Information")
    print("="*60)
    
    try:
        # Initialize translator
        translator = create_speech_translator(initialize=False)
        
        # Get system information
        info = translator.get_system_info()
        
        print("🖥️  System Configuration:")
        for key, value in info['configuration'].items():
            print(f"  {key}: {value}")
        
        print("\\n🔧 Component Status:")
        for component, loaded in info['components_loaded'].items():
            status = "✅ Loaded" if loaded else "❌ Not Loaded"
            print(f"  {component}: {status}")
        
        print(f"\\n🌍 Supported Languages: {info['supported_languages']}")
        print(f"👥 Registered Speakers: {info['registered_speakers']}")
        
        if any(info['statistics'].values()):
            print("\\n📊 Usage Statistics:")
            for key, value in info['statistics'].items():
                print(f"  {key}: {value}")
    
    except Exception as e:
        print(f"💥 Error: {str(e)}")
        logger.exception("Example 6 failed")


def setup_sample_files():
    """
    Setup sample directories and create placeholder files
    """
    print("\\n" + "="*60)
    print("Setting up Sample Directories")
    print("="*60)
    
    # Create directories
    directories = [
        "data/samples",
        "data/voice_samples"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Create placeholder README files
    samples_readme = Path("data/samples/README.md")
    samples_readme.write_text("""
# Sample Audio Files

Place your audio files here for testing:

- `input_speech.wav` - Input audio for translation
- `audio_1.wav`, `audio_2.wav`, etc. - Files for batch processing

Supported formats: WAV, MP3, M4A, FLAC, OGG
Recommended: 16kHz or higher sample rate
""")
    
    voice_samples_readme = Path("data/voice_samples/README.md")
    voice_samples_readme.write_text("""
# Voice Sample Files

Place voice sample files here for voice cloning:

- `target_voice.wav` - Main voice sample for cloning
- `john_sample_1.wav`, `john_sample_2.wav`, etc. - Multiple samples for better cloning

Requirements:
- Clear, high-quality audio
- At least 10 seconds total duration
- Same speaker across all samples
- Minimal background noise
""")
    
    print(f"📄 Created: {samples_readme}")
    print(f"📄 Created: {voice_samples_readme}")
    print("\\n✅ Sample directories setup complete!")
    print("\\n📋 Next steps:")
    print("1. Add your audio files to data/samples/")
    print("2. Add voice samples to data/voice_samples/")
    print("3. Run the examples again to test with real files")


def main():
    """
    Main function to run all examples
    """
    print("🎙️  Speech Translation System - Example Usage")
    print("=" * 70)
    
    # Setup sample directories first
    setup_sample_files()
    
    # Run examples
    examples = [
        example_1_basic_translation,
        example_2_text_to_speech,
        example_3_speaker_registration,
        example_4_batch_processing,
        example_5_component_usage,
        example_6_system_info
    ]
    
    for example in examples:
        try:
            example()
        except KeyboardInterrupt:
            print("\\n🛑 Interrupted by user")
            break
        except Exception as e:
            print(f"\\n💥 Example failed: {str(e)}")
            logger.exception(f"Failed to run {example.__name__}")
    
    print("\\n" + "="*70)
    print("🎯 Examples completed!")
    print("\\nFor more advanced usage, see:")
    print("- CLI: python -m src.ui.cli --help")
    print("- API documentation in docs/")
    print("- Test files in tests/")


if __name__ == "__main__":
    main()