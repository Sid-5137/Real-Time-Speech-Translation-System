# AI Speech Translation System

**Live Demo:** [https://huggingface.co/spaces/SidML/AI-Speech-Translator](https://huggingface.co/spaces/SidML/AI-Speech-Translator)

## Overview

An advanced AI-powered speech translation system that automatically detects spoken language, translates it to your target language, and generates natural-sounding speech output. Built with modern machine learning technologies including OpenAI Whisper, enhanced translation engines, and text-to-speech synthesis.

## Features

- **Automatic Language Detection** - Identifies source language from 12+ supported languages
- **High-Quality Speech Recognition** - OpenAI Whisper integration for accurate transcription
- **Smart Translation Engine** - Enhanced Hindi-English support with multi-language translation
- **Natural Voice Synthesis** - Multiple TTS engines with fallback support
- **Modern Web Interface** - Apple-style dark mode interface
- **Real-time Processing** - Fast audio processing with live status updates
- **Auto-Initialization** - System loads automatically without manual setup

## Quick Start

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Internet connection (for translation services)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Speech_Translation_System.git
cd Speech_Translation_System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python improved_speech_app.py
```

The web interface will be available at `http://localhost:7863`

## How to Use

1. **Upload Audio** - Upload an audio file (WAV, MP3, M4A) or record directly
2. **Select Target Language** - Choose your desired translation language
3. **Get Results** - View original transcription, translated text, and listen to synthesized speech

## Supported Languages

| Language | Recognition | Translation | Speech Synthesis |
|----------|-------------|-------------|------------------|
| English | Yes | Yes | Yes |
| Spanish | Yes | Yes | Yes |
| French | Yes | Yes | Yes |
| German | Yes | Yes | Yes |
| Italian | Yes | Yes | Yes |
| Portuguese | Yes | Yes | Yes |
| Russian | Yes | Yes | Yes |
| Japanese | Yes | Yes | Yes |
| Korean | Yes | Yes | Yes |
| Chinese | Yes | Yes | Yes |
| Arabic | Yes | Yes | Yes |

## Architecture

```
Audio Input → Whisper (Speech Recognition) → Translation Engine → TTS → Audio Output
                ↓                              ↓                      ↓
        Auto Language Detection        Enhanced Dictionary        Natural Voice
```

### Core Components

- **Speech Recognition**: OpenAI Whisper 'small' model optimized for Hindi
- **Translation**: Multi-tier system (Enhanced Dictionary → MyMemory API → Fallback)
- **Language Detection**: Script-based detection for 12+ languages
- **Text-to-Speech**: Google TTS → pyttsx3 → Mock audio (fallback chain)
- **Web Interface**: Gradio with Apple-style dark mode

## Project Structure

```
Speech_Translation_System/
├── app.py                      # Deployment-ready app
├── improved_speech_app.py      # Local development app  
├── requirements.txt            # Dependencies
├── Dockerfile                  # Container deployment
│
├── src/                        # Core source code
│   ├── translation/            # Translation engines
│   │   ├── improved_translator.py # Enhanced Hindi-English
│   │   └── simple_translator.py   # Basic translation
│   ├── tts/                    # Text-to-speech
│   │   └── tts_service.py      # TTS service
│   ├── speech_recognition/     # Speech processing
│   ├── audio_processing/       # Audio utilities
│   └── pipeline/               # Main pipeline
│
├── tests/                      # Test suite
├── examples/                   # Usage examples
└── data/                       # Sample data
```

## Technical Details

### Dependencies

- **OpenAI Whisper** - Speech recognition
- **Gradio 4.0+** - Web interface
- **LibROSA** - Audio processing  
- **gTTS / pyttsx3** - Text-to-speech
- **NumPy / SciPy** - Numerical computing
- **Requests** - API communication

### Performance
- **Processing Speed**: ~15-25 seconds for 10-second audio
- **Memory Usage**: ~2GB RAM (with Whisper 'small' model)
- **Concurrent Users**: Supports multiple simultaneous requests

## Docker Deployment

```bash
# Build the Docker image
docker build -t ai-speech-translator .

# Run the container
docker run -p 7860:7860 ai-speech-translator
```

## Development

### Running Tests
```bash
python -m pytest tests/ -v
```

### Core Pipeline Usage
```python
from src.translation.improved_translator import create_improved_translator
from src.tts.tts_service import create_tts_service

translator = create_improved_translator()
tts = create_tts_service()
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test category
python -m pytest tests/test_audio_processing.py -v
python -m pytest tests/test_pipeline.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 📚 Documentation

- 📖 **[User Guide](docs/user_guide.md)** - Comprehensive usage instructions
- 🏗️ **[Architecture Guide](docs/architecture.md)** - Technical architecture details
- 💡 **[Examples](examples/usage_examples.py)** - Code examples and tutorials

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI Whisper** - Speech recognition
- **Google Translate** - Translation services
- **Coqui TTS** - Text-to-speech and voice cloning
- **Hugging Face Transformers** - Local translation models
- **PyTorch** - Deep learning framework

## 🔗 Links

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Coqui TTS](https://github.com/coqui-ai/TTS)
- [Google Translate API](https://cloud.google.com/translate)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

---

**🎉 Ready to get started? Run `python setup_system.py` to begin!**