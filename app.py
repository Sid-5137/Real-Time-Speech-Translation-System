#!/usr/bin/env python3
"""
AI Speech Translation System - Deployment Version
Optimized for Hugging Face Spaces deployment

Features:
- Real-time speech recognition with Whisper
- Auto language detection for 12+ languages  
"""

import gradio as gr
import sys
import os
import time
import tempfile
import threading
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import soundfile as sf

# Add src to Python path for local imports
current_dir = Path(__file__).parent
src_path = current_dir / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

# Import with error handling for deployment
try:
    import whisper
    import librosa
    WHISPER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Whisper not available: {e}")
    WHISPER_AVAILABLE = False

try:
    from translation.improved_translator import create_improved_translator
    from tts.tts_service import create_tts_service
    SERVICES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Services not available: {e}")
    SERVICES_AVAILABLE = False


class DeploymentSpeechApp:
    """Production-ready speech translation app"""
    
    def __init__(self):
        self.whisper_model = None
        self.translator = None
        self.tts_service = None
        self.initialization_status = "🔄 Initializing system..."
        self.system_ready = False
        
        # Language options
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
        
        self.temp_dir = Path(tempfile.gettempdir()) / "speech_translation_deploy"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Start initialization
        self._start_initialization()
    
    def _start_initialization(self):
        """Initialize system components"""
        def init_worker():
            try:
                if not WHISPER_AVAILABLE or not SERVICES_AVAILABLE:
                    self.initialization_status = "❌ Missing dependencies for full functionality"
                    return
                    
                self.initialization_status = "🎙️ Loading speech recognition..."
                self.whisper_model = whisper.load_model("small")
                
                self.initialization_status = "🌍 Setting up translation..."
                self.translator = create_improved_translator()
                
                self.initialization_status = "🎵 Preparing text-to-speech..."
                self.tts_service = create_tts_service()
                
                self.initialization_status = "✅ System ready!"
                self.system_ready = True
                
            except Exception as e:
                self.initialization_status = f"❌ Initialization failed: {str(e)}"
                self.system_ready = False
        
        threading.Thread(target=init_worker, daemon=True).start()
    
    def get_system_status(self) -> str:
        return self.initialization_status
    
    def process_audio(
        self, 
        audio_file: str, 
        target_lang: str = "en"
    ) -> Tuple[str, str, str, Optional[str], str]:
        """Process audio file and return results"""
        
        if not self.system_ready:
            status = f"⏳ System not ready. Status: {self.initialization_status}"
            return "", "", "", None, status
        
        if audio_file is None:
            return "", "", "", None, "❌ Please upload an audio file"
        
        try:
            start_time = time.time()
            
            # Step 1: Transcribe
            result = self.whisper_model.transcribe(
                audio_file,
                task="transcribe",
                verbose=False
            )
            
            transcription = result['text'].strip()
            detected_lang = result.get('language', 'unknown')
            
            if not transcription:
                return "", "", detected_lang, None, "❌ No speech detected"
            
            # Step 2: Translate
            if target_lang == "auto":
                target_lang = "en" if detected_lang != "en" else "hi"
            
            translation_result = self.translator.translate_text(
                text=transcription,
                source_lang=detected_lang,
                target_lang=target_lang
            )
            
            if not translation_result['success']:
                return transcription, "", detected_lang, None, f"❌ Translation failed"
            
            translation = translation_result['translated_text']
            
            # Step 3: Generate speech
            timestamp = int(time.time())
            audio_filename = f"output_{timestamp}.wav"
            audio_output_path = self.temp_dir / audio_filename
            
            tts_result = self.tts_service.synthesize_speech(
                text=translation,
                language=target_lang,
                output_path=str(audio_output_path)
            )
            
            if not tts_result['success']:
                return transcription, translation, detected_lang, None, f"❌ TTS failed"
            
            audio_output = tts_result['audio_path']
            
            # Final status
            total_time = time.time() - start_time
            status = f"""
✅ **Translation Complete!**

**📊 Summary:**
- ⏱️ **Time:** {total_time:.1f}s
- 🌍 **From:** {detected_lang.upper()} → {target_lang.upper()}
- 🎵 **Engine:** {tts_result['engine']}
- 📈 **Service:** {translation_result.get('service', 'Unknown')}
            """
            
            return transcription, translation, detected_lang, audio_output, status
            
        except Exception as e:
            return "", "", "", None, f"❌ Error: {str(e)}"
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        # Enhanced CSS for production
        css = """
        /* Production-ready Apple Dark Mode */
        .gradio-container {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            background: #000000;
            color: #ffffff;
        }
        
        body {
            background: #000000 !important;
            color: #ffffff !important;
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
        
        .status-box {
            background: linear-gradient(135deg, #007aff 0%, #5856d6 100%);
            color: #ffffff;
            padding: 16px;
            border-radius: 12px;
            text-align: center;
            margin: 16px 0;
            font-weight: 500;
        }
        
        /* Force dark mode for all components */
        .gradio-container * {
            background-color: #1c1c1e !important;
            color: #ffffff !important;
        }
        
        .gradio-container .gr-button {
            background: #007aff !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
        }
        
        .gradio-container .gr-button:hover {
            background: #0a84ff !important;
        }
        
        .gradio-container .gr-textbox, 
        .gradio-container .gr-textbox input,
        .gradio-container .gr-textbox textarea {
            background: #2c2c2e !important;
            border: 1px solid #48484a !important;
            color: #ffffff !important;
            border-radius: 8px !important;
        }
        
        .gradio-container .gr-dropdown,
        .gradio-container .gr-dropdown select {
            background: #2c2c2e !important;
            border: 1px solid #48484a !important;
            color: #ffffff !important;
            border-radius: 8px !important;
        }
        """
        
        with gr.Blocks(css=css, title="AI Speech Translation System") as interface:
            
            # Header
            gr.HTML("""
            <div class="header-gradient">
                <h1 style="font-size: 2.5em; margin: 0; font-weight: 700;">🎙️ AI Speech Translator</h1>
                <p style="font-size: 1.2em; margin: 16px 0 0 0; opacity: 0.8;">
                    Real-time Speech Translation • Auto Language Detection • 12+ Languages
                </p>
                <p style="font-size: 1em; margin: 8px 0 0 0; opacity: 0.6;">
                    Upload audio → Automatic transcription → Smart translation → Natural speech output
                </p>
            </div>
            """)
            
            # Status display
            with gr.Row():
                status_display = gr.Markdown(
                    value=f"**{self.get_system_status()}**",
                    elem_classes=["status-box"]
                )
            
            # Main interface
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 Upload & Configure")
                    
                    audio_input = gr.Audio(
                        label="🎤 Upload Audio or Record",
                        type="filepath",
                        sources=["upload", "microphone"]
                    )
                    
                    target_lang = gr.Dropdown(
                        choices=list(self.languages.keys()),
                        value="en",
                        label="🎯 Target Language"
                    )
                    
                    process_btn = gr.Button("🚀 Translate Audio", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📋 Results")
                    
                    detected_lang_display = gr.Textbox(
                        label="🔍 Detected Language",
                        interactive=False
                    )
                    
                    transcription_output = gr.Textbox(
                        label="📝 Original Text",
                        lines=3
                    )
                    
                    translation_output = gr.Textbox(
                        label="🌍 Translated Text", 
                        lines=3
                    )
                    
                    audio_output = gr.Audio(label="🎵 Translated Speech")
            
            # Detailed status
            detailed_status = gr.Markdown(
                value="Upload an audio file and click 'Translate Audio' to start..."
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
            
            # Tips section
            with gr.Accordion("💡 How to Use", open=False):
                gr.Markdown("""
                ### 🎯 Quick Start
                1. **Upload** an audio file (WAV, MP3, M4A) or record directly
                2. **Select** your target language (or keep "Auto-detect")
                3. **Click** "Translate Audio" 
                4. **Listen** to the results!
                
                ### ✨ Features
                - 🔍 **Auto Language Detection** - Automatically detects 12+ languages
                - 🎯 **Enhanced Hindi Support** - Optimized for Hindi-English translation
                - 🎵 **Natural Speech Output** - High-quality text-to-speech synthesis
                - 🌙 **Beautiful UI** - Apple-inspired dark mode design
                
                ### 🌍 Supported Languages
                Hindi, English, Spanish, French, German, Italian, Portuguese, Russian, Japanese, Korean, Chinese, Arabic
                
                ### 🏗️ Tech Stack
                - **Speech Recognition**: OpenAI Whisper
                - **Translation**: Enhanced algorithms + API fallbacks
                - **Speech Synthesis**: Google TTS + offline engines
                - **Interface**: Gradio with custom styling
                """)
            
            # Footer
            gr.HTML("""
            <div style="text-align: center; margin-top: 32px; padding: 24px; background: #1c1c1e; border-radius: 12px;">
                <p style="color: #98989d; margin: 0; font-size: 14px;">
                    🎉 AI Speech Translation System • Built with Whisper, Gradio & Modern ML
                </p>
            </div>
            """)
        
        return interface


def main():
    """Launch the application"""
    print("🚀 Starting AI Speech Translation System...")
    print("🌟 Deployment-ready version for cloud hosting")
    
    app = DeploymentSpeechApp()
    interface = app.create_interface()
    
    # Launch configuration for deployment
    interface.launch(
        server_name="0.0.0.0",  # Listen on all interfaces for cloud deployment
        server_port=7860,       # Standard port for Hugging Face Spaces
        share=False,
        debug=False,
        show_api=False,
        inbrowser=False  # Don't auto-open browser in cloud
    )


if __name__ == "__main__":
    main()