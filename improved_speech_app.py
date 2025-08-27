#!/usr/bin/env python3
import gradio as gr
import sys
import os
import time
import tempfile
import asyncio
import threading
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import soundfile as sf

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import components
try:
    import whisper
    import librosa
    # Direct import to avoid dependency issues
    sys.path.insert(0, str(Path(__file__).parent / "src" / "translation"))
    from improved_translator import create_improved_translator
    sys.path.insert(0, str(Path(__file__).parent / "src" / "tts"))
    from tts_service import create_tts_service
    DEPENDENCIES_OK = True
except ImportError as e:
    DEPENDENCIES_OK = False
    print(f"⚠️ Dependencies not available: {e}")


class AutoInitSpeechApp:
    """Auto-initializing speech translation app with improved UX"""
    
    def __init__(self):
        # Initialize immediately
        self.whisper_model = None
        self.translator = None
        self.tts_service = None
        self.initialization_status = "🔄 Initializing system..."
        self.system_ready = False
        
        # Language options with auto-detect
        self.languages = {
            "auto": "🔍 Auto-detect",
            "hi": "🇮🇳 Hindi",
            "en": "🇺🇸 English",
            "es": "🇪🇸 Spanish", 
            "fr": "🇫🇷 French",
            "de": "🇩🇪 German",
            "it": "🇮🇹 Italian",
            "pt": "🇵🇹 Portuguese",
            "ru": "🇷🇺 Russian",
            "ja": "🇯🇵 Japanese",
            "ko": "🇰🇷 Korean",
            "zh": "🇨🇳 Chinese",
            "ar": "🇸🇦 Arabic"
        }
        
        self.temp_dir = Path(tempfile.gettempdir()) / "improved_speech_app"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Start auto-initialization in background
        self._start_auto_initialization()
    
    def _start_auto_initialization(self):
        """Start background initialization"""
        def init_worker():
            try:
                if not DEPENDENCIES_OK:
                    self.initialization_status = "❌ Missing dependencies. Please install: pip install openai-whisper librosa gtts pydub"
                    return
                    
                self.initialization_status = "🎙️ Loading speech recognition model..."
                time.sleep(0.5)  # Small delay for UI update
                
                self.whisper_model = whisper.load_model("small")
                self.initialization_status = "✅ Speech recognition loaded! Setting up translation..."
                time.sleep(0.5)
                
                self.translator = create_improved_translator()
                self.initialization_status = "✅ Translation ready! Preparing text-to-speech..."
                time.sleep(0.5)
                
                self.tts_service = create_tts_service()
                self.initialization_status = "🎉 System fully initialized and ready to use!"
                
                self.system_ready = True
                
            except Exception as e:
                self.initialization_status = f"❌ Initialization failed: {str(e)}. Please check dependencies."
                self.system_ready = False
        
        # Start in background thread
        threading.Thread(target=init_worker, daemon=True).start()
    
    def get_system_status(self) -> str:
        """Get current system status"""
        return self.initialization_status
    
    def process_audio(
        self, 
        audio_file: str, 
        target_lang: str = "en"
    ) -> Tuple[str, str, str, Optional[str], str]:
        """
        Process audio with auto language detection
        
        Returns: (transcription, translation, detected_lang, audio_output, status)
        """
        
        if not self.system_ready:
            status = f"⏳ System not ready yet. Status: {self.initialization_status}"
            return "", "", "", None, status
        
        if audio_file is None:
            return "", "", "", None, "❌ Please upload an audio file"
        
        try:
            start_time = time.time()
            
            # Step 1: Transcribe with auto language detection
            status = "🎙️ Transcribing audio and detecting language..."
            
            result = self.whisper_model.transcribe(
                audio_file,
                task="transcribe",
                verbose=False
            )
            
            transcription = result['text'].strip()
            detected_lang = result.get('language', 'unknown')
            
            if not transcription:
                return "", "", detected_lang, None, "❌ No speech detected in audio"
            
            # Step 2: Auto-translate if needed
            if target_lang == "auto":
                # If auto target, translate to English if not English
                target_lang = "en" if detected_lang != "en" else "hi"
            
            status = f"🔄 Translating from {detected_lang} to {target_lang}..."
            
            translation_result = self.translator.translate_text(
                text=transcription,
                source_lang=detected_lang,
                target_lang=target_lang
            )
            
            if not translation_result['success']:
                return transcription, "", detected_lang, None, f"❌ Translation failed: {translation_result.get('error')}"
            
            translation = translation_result['translated_text']
            
            # Step 3: Generate speech
            status = "🎵 Generating translated speech..."
            
            # Generate unique output filename in a web-accessible location
            timestamp = int(time.time())
            audio_filename = f"translated_speech_{timestamp}.wav"
            audio_output_path = self.temp_dir / audio_filename
            
            tts_result = self.tts_service.synthesize_speech(
                text=translation,
                language=target_lang,
                output_path=str(audio_output_path)
            )
            
            if not tts_result['success']:
                return transcription, translation, detected_lang, None, f"❌ TTS failed: {tts_result.get('error')}"
            
            # Verify audio file exists and is accessible
            audio_output = tts_result['audio_path']
            if not audio_output or not Path(audio_output).exists():
                return transcription, translation, detected_lang, None, "❌ Audio file was not generated properly"
            
            # Ensure the audio file is accessible to Gradio
            # Make sure the path is absolute and exists
            audio_output = str(Path(audio_output).resolve())
            
            # Final status
            total_time = time.time() - start_time
            final_status = f"""
✅ **Translation Complete!**

**📊 Processing Summary:**
- ⏱️ **Total Time:** {total_time:.1f} seconds
- 🌍 **Detected Language:** {detected_lang.upper()} ({self.languages.get(detected_lang, 'Unknown')})
- 🎯 **Target Language:** {target_lang.upper()} ({self.languages.get(target_lang, 'Unknown')})
- 🎵 **Audio Engine:** {tts_result['engine']}
- 📈 **Translation Confidence:** {translation_result.get('confidence', 'N/A')}

**🔧 Services Used:**
- Speech Recognition: Whisper 'small' model
- Translation: {translation_result.get('service', 'Unknown')}
- Text-to-Speech: {tts_result['engine']}
            """
            
            return transcription, translation, detected_lang, audio_output, final_status
            
        except Exception as e:
            error_status = f"❌ Processing failed: {str(e)}"
            return "", "", "", None, error_status
    
    def create_interface(self):
        """Create improved Gradio interface"""
        
        # Apple-style dark mode CSS with improved readability
        css = """
        /* Main container with Apple dark theme */
        .gradio-container {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            background: #000000;
            color: #ffffff;
        }
        
        /* Override Gradio's default styles for better Apple dark mode */
        body {
            background: #000000 !important;
            color: #ffffff !important;
        }
        
        .main-container {
            background: #1c1c1e;
            border-radius: 16px;
            padding: 24px;
            margin: 16px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            border: 1px solid #38383a;
        }
        
        .status-box {
            background: linear-gradient(135deg, #007aff 0%, #5856d6 100%);
            color: #ffffff;
            padding: 16px;
            border-radius: 12px;
            text-align: center;
            margin: 16px 0;
            font-weight: 500;
            font-size: 15px;
        }
        
        .result-box {
            background: #2c2c2e;
            border: 1px solid #48484a;
            border-radius: 12px;
            padding: 16px;
            margin: 12px 0;
            color: #ffffff;
        }
        
        .upload-area {
            border: 2px dashed #007aff;
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            background: #1c1c1e;
            transition: all 0.3s ease;
            color: #ffffff;
        }
        
        .upload-area:hover {
            background: #2c2c2e;
            border-color: #0a84ff;
        }
        
        .header-gradient {
            background: linear-gradient(135deg, #1d1d1f 0%, #2c2c2e 100%);
            color: #ffffff;
            padding: 32px;
            border-radius: 16px;
            margin-bottom: 24px;
            text-align: center;
            border: 1px solid #48484a;
        }
        
        .section-header {
            color: #ffffff;
            font-weight: 600;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 1px solid #48484a;
        }
        
        /* Force Apple dark mode for all Gradio components */
        .gradio-container, .gradio-container * {
            background-color: #1c1c1e !important;
            color: #ffffff !important;
        }
        
        /* Buttons with Apple blue */
        .gradio-container .gr-button {
            background: #007aff !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
        }
        
        .gradio-container .gr-button:hover {
            background: #0a84ff !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 122, 255, 0.3) !important;
        }
        
        /* Input fields */
        .gradio-container .gr-textbox, 
        .gradio-container .gr-textbox input,
        .gradio-container .gr-textbox textarea {
            background: #2c2c2e !important;
            border: 1px solid #48484a !important;
            color: #ffffff !important;
            border-radius: 8px !important;
        }
        
        /* Dropdown menus */
        .gradio-container .gr-dropdown,
        .gradio-container .gr-dropdown .gr-box,
        .gradio-container .gr-dropdown select {
            background: #2c2c2e !important;
            border: 1px solid #48484a !important;
            color: #ffffff !important;
            border-radius: 8px !important;
        }
        
        /* Audio components */
        .gradio-container .gr-audio {
            background: #2c2c2e !important;
            border: 1px solid #48484a !important;
            border-radius: 12px !important;
        }
        
        /* Labels and markdown */
        .gradio-container .gr-form label,
        .gradio-container .markdown,
        .gradio-container .gr-markdown,
        .gradio-container p,
        .gradio-container span {
            color: #ffffff !important;
        }
        
        /* File upload area */
        .gradio-container .gr-file-upload {
            background: #2c2c2e !important;
            border: 2px dashed #007aff !important;
            border-radius: 12px !important;
            color: #ffffff !important;
        }
        
        /* Accordion */
        .gradio-container .gr-accordion {
            background: #2c2c2e !important;
            border: 1px solid #48484a !important;
            border-radius: 12px !important;
        }
        
        /* Progress bar */
        .gradio-container .gr-progress {
            background: #2c2c2e !important;
            border-radius: 8px !important;
        }
        
        /* Ensure all text is readable */
        .gradio-container * {
            color: #ffffff !important;
        }
        
        /* Special handling for code blocks */
        .gradio-container pre, .gradio-container code {
            background: #1c1c1e !important;
            border: 1px solid #48484a !important;
            color: #ffffff !important;
        }
        """
        
        with gr.Blocks(css=css, title="Improved Speech Translation System") as interface:
            
            # Header with Apple-style dark design
            gr.HTML("""
            <div class="header-gradient">
                <h1 style="font-size: 2.5em; margin: 0; font-weight: 700; color: #ffffff;">🎙️ AI Speech Translator</h1>
                <p style="font-size: 1.2em; margin: 16px 0 0 0; opacity: 0.8; color: #ffffff;">
                    Auto-detect • Accurate Translation • Natural Voice • Real-time
                </p>
            </div>
            """)
            
            # Auto-updating status with better visibility
            with gr.Row():
                with gr.Column():
                    status_display = gr.Markdown(
                        value=f"**{self.get_system_status()}**",
                        elem_classes=["status-box"],
                        visible=True
                    )
            
            # Main interface
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<h3 class="section-header">📤 Upload & Configure</h3>')
                    
                    # Audio upload with improved styling
                    audio_input = gr.Audio(
                        label="🎤 Upload Audio or Record",
                        type="filepath",
                        sources=["upload", "microphone"],
                        elem_classes=["upload-area"]
                    )
                    
                    # Target language selection
                    target_lang = gr.Dropdown(
                        choices=list(self.languages.keys()),
                        value="en",
                        label="🎯 Target Language",
                        info="Language to translate to"
                    )
                    
                    # Process button
                    process_btn = gr.Button(
                        "🚀 Translate Audio",
                        variant="primary",
                        size="lg",
                        elem_id="process-btn"
                    )
                
                with gr.Column(scale=1):
                    gr.HTML('<h3 class="section-header">📤 Results</h3>')
                    
                    # Detected language display
                    detected_lang_display = gr.Textbox(
                        label="🔍 Detected Language",
                        interactive=False,
                        placeholder="Upload audio to see detected language..."
                    )
                    
                    # Transcription
                    transcription_output = gr.Textbox(
                        label="📝 Original Text",
                        lines=3,
                        placeholder="Original speech will appear here...",
                        elem_classes=["result-box"]
                    )
                    
                    # Translation
                    translation_output = gr.Textbox(
                        label="🌍 Translated Text",
                        lines=3,
                        placeholder="Translation will appear here...",
                        elem_classes=["result-box"]
                    )
                    
                    # Audio output
                    audio_output = gr.Audio(
                        label="🎵 Translated Speech",
                        elem_classes=["result-box"]
                    )
            
            # Detailed status
            detailed_status = gr.Markdown(
                value="Upload an audio file and click 'Translate Audio' to start...",
                elem_classes=["result-box"]
            )
            
            # Event handlers
            process_btn.click(
                self.process_audio,
                inputs=[audio_input, target_lang],
                outputs=[
                    transcription_output,
                    translation_output, 
                    detected_lang_display,
                    audio_output,
                    detailed_status
                ]
            )
            
            # Status refresh on audio upload and process button click
            def refresh_status():
                return f"**{self.get_system_status()}**"
            
            audio_input.change(
                refresh_status,
                outputs=status_display
            )
            
            process_btn.click(
                refresh_status,
                outputs=status_display
            )
            
            # Examples and tips
            with gr.Accordion("💡 Tips & Examples", open=False):
                gr.Markdown("""
                ### 🎯 Quick Start Guide
                
                1. **Upload Audio**: Use the microphone or upload a file (WAV, MP3, M4A)
                2. **Select Target**: Choose your desired translation language
                3. **Click Translate**: The system automatically detects source language and translates
                4. **Get Results**: View transcription, translation, and listen to synthesized speech
                
                ### ✨ Key Features
                
                - **🔍 Auto Language Detection**: Automatically detects Hindi, English, Spanish, and more
                - **🎯 Improved Hindi Translation**: Enhanced accuracy for Hindi ↔ English translations
                - **🚀 Auto-Initialization**: System loads automatically in background
                - **🎵 Multiple TTS Engines**: Google TTS, offline TTS, and fallback options
                - **⚡ Real-time Feedback**: Live status updates during processing
                
                ### 📝 Example Phrases
                
                **Hindi Examples:**
                - "नमस्ते, आप कैसे हैं?" → "Hello, how are you?"
                - "धन्यवाद" → "Thank you"
                - "जब मैं छोटा था" → "When I was small"
                
                **English Examples:**
                - "Hello, how are you?" → "नमस्ते, आप कैसे हैं?"
                - "Thank you very much" → "बहुत धन्यवाद"
                
                ### 🔧 Technical Details
                
                - **Speech Recognition**: OpenAI Whisper 'small' model (optimized for Hindi)
                - **Translation**: Enhanced dictionary + MyMemory API + fallbacks
                - **Audio Generation**: Google TTS, pyttsx3, mock audio generation
                - **Auto-Detection**: Script-based language detection for 12+ languages
                """)
            
            # Footer with dark mode styling
            gr.HTML("""
            <div style="text-align: center; margin-top: 32px; padding: 24px; background: #1c1c1e; border: 1px solid #48484a; border-radius: 12px;">
                <h4 style="color: #ffffff; margin: 0; font-weight: 600;">🎉 Enhanced Speech Translation System</h4>
                <p style="color: #98989d; margin: 12px 0 0 0; font-size: 14px;">
                    Auto-detect • Improved Hindi Support • Dark Mode • Auto-initialization
                </p>
            </div>
            """)
        
        return interface


def main():
    """Launch the improved application"""
    print("🚀 Launching Improved Speech Translation System...")
    print("✨ Features: Auto-init • Auto-detect • Better Hindi translation • Improved UI")
    print("🌐 Opening enhanced web interface...")
    
    app = AutoInitSpeechApp()
    interface = app.create_interface()
    
    # Launch with better configuration
    interface.launch(
        server_name="localhost",
        server_port=7863,  # New port
        share=False,  # Set to True for public access
        debug=False,
        show_api=False,
        inbrowser=True,
        favicon_path=None,
        auth=None
    )


if __name__ == "__main__":
    main()