"""
Command Line Interface for Speech Translation System

This module provides a user-friendly CLI for the speech translation system.
"""

import click
import logging
import sys
from pathlib import Path
from typing import Optional, List
import json

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from ..pipeline.main_pipeline import create_speech_translator, SpeechTranslator
from ..config import SUPPORTED_LANGUAGES, WHISPER_MODEL_SIZE, DEFAULT_TRANSLATION_SERVICE, TTS_MODEL


# Initialize rich console
console = Console()


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('speech_translation.log'),
            logging.StreamHandler()
        ]
    )


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, verbose):
    """Speech Translation System with Voice Cloning"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    setup_logging(verbose)


@cli.command()
@click.argument('input_audio', type=click.Path(exists=True))
@click.argument('voice_sample', type=click.Path(exists=True))
@click.option('--source-lang', '-s', help='Source language code (auto-detect if not specified)')
@click.option('--target-lang', '-t', default='en', help='Target language code (default: en)')
@click.option('--output', '-o', type=click.Path(), help='Output audio file path')
@click.option('--speech-model', default=WHISPER_MODEL_SIZE, 
              help=f'Whisper model size (default: {WHISPER_MODEL_SIZE})')
@click.option('--translation-engine', default=DEFAULT_TRANSLATION_SERVICE,
              type=click.Choice(['google', 'local']),
              help=f'Translation engine (default: {DEFAULT_TRANSLATION_SERVICE})')
@click.option('--tts-model', default=TTS_MODEL, help=f'TTS model (default: {TTS_MODEL})')
@click.option('--device', default='auto', help='Device to use (auto, cpu, cuda)')
@click.pass_context
def translate(ctx, input_audio, voice_sample, source_lang, target_lang, output, 
             speech_model, translation_engine, tts_model, device):
    """Translate audio file with voice cloning."""
    
    try:
        # Validate language codes
        if target_lang not in SUPPORTED_LANGUAGES:
            console.print(f"[red]Error: Unsupported target language '{target_lang}'[/red]")
            console.print("Supported languages:", list(SUPPORTED_LANGUAGES.keys()))
            sys.exit(1)
        
        if source_lang and source_lang not in SUPPORTED_LANGUAGES:
            console.print(f"[red]Error: Unsupported source language '{source_lang}'[/red]")
            sys.exit(1)
        
        # Generate output path if not provided
        if not output:
            input_path = Path(input_audio)
            output = input_path.parent / f"{input_path.stem}_translated_{target_lang}.wav"
        
        console.print(Panel.fit(f"🎙️  Speech Translation System", style="bold blue"))
        console.print(f"📁 Input: {input_audio}")
        console.print(f"🎯 Voice Sample: {voice_sample}")
        console.print(f"🌍 Translation: {source_lang or 'auto'} → {target_lang}")
        console.print(f"💾 Output: {output}")
        
        # Progress tracking
        progress_messages = []
        def progress_callback(message):
            progress_messages.append(message)
            console.print(f"⏳ {message}")
        
        # Initialize translator
        console.print("\\n🚀 Initializing translation system...")
        translator = create_speech_translator(
            speech_model=speech_model,
            translation_engine=translation_engine,
            tts_model=tts_model,
            device=device,
            initialize=False
        )
        
        translator.progress_callback = progress_callback
        translator.initialize()
        
        # Perform translation
        console.print("\\n🔄 Starting translation process...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            
            task = progress.add_task("Translating...", total=100)
            
            result = translator.translate_audio(
                input_audio=input_audio,
                source_lang=source_lang,
                target_lang=target_lang,
                voice_sample=voice_sample,
                output_path=output,
                return_intermediate=True
            )
        
        # Display results
        if result['success']:
            console.print("\\n✅ [green]Translation completed successfully![/green]")
            
            # Create results table
            table = Table(title="Translation Results")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="white")
            
            table.add_row("Original Text", result['original_text'][:100] + "..." if len(result['original_text']) > 100 else result['original_text'])
            table.add_row("Translated Text", result['translated_text'][:100] + "..." if len(result['translated_text']) > 100 else result['translated_text'])
            table.add_row("Source Language", result['source_language'])
            table.add_row("Target Language", result['target_language'])
            table.add_row("Processing Time", f"{result['processing_time']:.2f} seconds")
            table.add_row("Audio Duration", f"{result['audio_duration']:.2f} seconds")
            table.add_row("Output File", str(result['output_audio']))
            
            console.print(table)
            
        else:
            console.print(f"\\n❌ [red]Translation failed: {result['error']}[/red]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"\\n💥 [red]Unexpected error: {str(e)}[/red]")
        if ctx.obj['verbose']:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.argument('text')
@click.argument('voice_sample', type=click.Path(exists=True))
@click.option('--source-lang', '-s', required=True, help='Source language code')
@click.option('--target-lang', '-t', default='en', help='Target language code')
@click.option('--output', '-o', type=click.Path(), help='Output audio file path')
@click.option('--tts-model', default=TTS_MODEL, help=f'TTS model (default: {TTS_MODEL})')
@click.option('--device', default='auto', help='Device to use (auto, cpu, cuda)')
def text_to_speech(text, voice_sample, source_lang, target_lang, output, tts_model, device):
    """Translate text and generate speech with voice cloning."""
    
    try:
        # Validate inputs
        if not output:
            output = f"translated_speech_{target_lang}.wav"
        
        console.print(Panel.fit("📝 Text to Speech Translation", style="bold green"))
        console.print(f"📝 Text: {text}")
        console.print(f"🎯 Voice Sample: {voice_sample}")
        console.print(f"🌍 Translation: {source_lang} → {target_lang}")
        
        # Initialize translator
        translator = create_speech_translator(tts_model=tts_model, device=device)
        
        # Perform translation and speech generation
        result = translator.translate_text_with_voice(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            voice_sample=voice_sample,
            output_path=output
        )
        
        if result['success']:
            console.print("\\n✅ [green]Text translation completed![/green]")
            console.print(f"🎵 Audio saved to: {result['output_audio']}")
        else:
            console.print(f"\\n❌ [red]Translation failed: {result['error']}[/red]")
            
    except Exception as e:
        console.print(f"\\n💥 [red]Error: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('audio_files', nargs=-1, required=True)
@click.argument('voice_sample', type=click.Path(exists=True))
@click.option('--target-lang', '-t', default='en', help='Target language code')
@click.option('--output-dir', '-d', type=click.Path(), help='Output directory')
@click.option('--speech-model', default=WHISPER_MODEL_SIZE, help='Whisper model size')
@click.option('--device', default='auto', help='Device to use')
def batch(audio_files, voice_sample, target_lang, output_dir, speech_model, device):
    """Batch translate multiple audio files."""
    
    try:
        if not output_dir:
            output_dir = Path.cwd() / "translated_batch"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        console.print(Panel.fit("📦 Batch Translation", style="bold yellow"))
        console.print(f"📁 Files: {len(audio_files)} audio files")
        console.print(f"🎯 Voice Sample: {voice_sample}")
        console.print(f"🌍 Target Language: {target_lang}")
        console.print(f"💾 Output Directory: {output_dir}")
        
        # Initialize translator
        translator = create_speech_translator(speech_model=speech_model, device=device)
        
        # Perform batch translation
        with Progress(console=console) as progress:
            task = progress.add_task("Processing batch...", total=len(audio_files))
            
            result = translator.batch_translate_audio(
                audio_files=list(audio_files),
                target_lang=target_lang,
                voice_sample=voice_sample,
                output_dir=output_dir
            )
            
            progress.update(task, completed=len(audio_files))
        
        # Display results
        console.print(f"\\n📊 Batch processing completed!")
        console.print(f"✅ Successful: {result['successful']}")
        console.print(f"❌ Failed: {result['failed']}")
        
        if result['failed_files']:
            console.print("\\n🚨 Failed files:")
            for failed in result['failed_files']:
                console.print(f"  - {failed['file']}: {failed['error']}")
                
    except Exception as e:
        console.print(f"\\n💥 [red]Error: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('speaker_name')
@click.argument('voice_samples', nargs=-1, required=True)
@click.option('--session-dir', type=click.Path(), help='Session directory to save speaker')
def register_speaker(speaker_name, voice_samples, session_dir):
    """Register a speaker voice for reuse."""
    
    try:
        console.print(Panel.fit(f"🎤 Registering Speaker: {speaker_name}", style="bold purple"))
        
        # Initialize voice cloner
        from ..voice_cloning.voice_cloner import create_voice_cloner
        cloner = create_voice_cloner()
        
        # Register speaker
        result = cloner.register_voice(speaker_name, list(voice_samples))
        
        console.print("\\n✅ [green]Speaker registered successfully![/green]")
        console.print(f"👤 Speaker: {result['speaker_name']}")
        console.print(f"🎵 Samples: {result['num_samples']}")
        console.print(f"⏱️  Duration: {result['total_duration']:.1f} seconds")
        
        # Save to session if specified
        if session_dir:
            session_path = Path(session_dir)
            cloner.save_speaker_data(session_path)
            console.print(f"💾 Saved to session: {session_path}")
            
    except Exception as e:
        console.print(f"\\n💥 [red]Error: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
def languages():
    """List supported languages."""
    
    console.print(Panel.fit("🌍 Supported Languages", style="bold blue"))
    
    table = Table()
    table.add_column("Code", style="cyan")
    table.add_column("Language", style="white")
    
    for code, name in SUPPORTED_LANGUAGES.items():
        table.add_row(code, name)
    
    console.print(table)


@cli.command()
@click.option('--speech-model', default=WHISPER_MODEL_SIZE, help='Speech model to check')
@click.option('--translation-engine', default=DEFAULT_TRANSLATION_SERVICE, help='Translation engine')
@click.option('--tts-model', default=TTS_MODEL, help='TTS model to check')
@click.option('--device', default='auto', help='Device to use')
def info(speech_model, translation_engine, tts_model, device):
    """Show system information and status."""
    
    try:
        console.print(Panel.fit("ℹ️  System Information", style="bold cyan"))
        
        # Create translator to get system info
        translator = create_speech_translator(
            speech_model=speech_model,
            translation_engine=translation_engine,
            tts_model=tts_model,
            device=device,
            initialize=False
        )
        
        info_data = translator.get_system_info()
        
        # Configuration table
        config_table = Table(title="Configuration")
        config_table.add_column("Component", style="cyan")
        config_table.add_column("Setting", style="white")
        
        for key, value in info_data['configuration'].items():
            config_table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(config_table)
        
        # Component status
        status_table = Table(title="Component Status")
        status_table.add_column("Component", style="cyan")
        status_table.add_column("Status", style="white")
        
        for component, loaded in info_data['components_loaded'].items():
            status = "✅ Loaded" if loaded else "❌ Not Loaded"
            status_table.add_row(component.replace('_', ' ').title(), status)
        
        console.print(status_table)
        
        # Statistics
        if any(info_data['statistics'].values()):
            stats_table = Table(title="Usage Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")
            
            for key, value in info_data['statistics'].items():
                stats_table.add_row(key.replace('_', ' ').title(), str(value))
            
            console.print(stats_table)
        
    except Exception as e:
        console.print(f"\\n💥 [red]Error getting system info: {str(e)}[/red]")


@cli.command()
@click.argument('session_path', type=click.Path())
def save_session(session_path):
    """Save current session including registered speakers."""
    try:
        # Create a basic translator and save session
        translator = create_speech_translator(initialize=False)
        translator.save_session(session_path)
        console.print(f"💾 Session saved to: {session_path}")
    except Exception as e:
        console.print(f"💥 [red]Error saving session: {str(e)}[/red]")


@cli.command()
@click.argument('session_path', type=click.Path(exists=True))
def load_session(session_path):
    """Load previous session."""
    try:
        translator = create_speech_translator(initialize=False)
        translator.load_session(session_path)
        console.print(f"📂 Session loaded from: {session_path}")
        
        # Show loaded speakers
        speakers = translator.get_registered_speakers()
        if speakers:
            console.print(f"👥 Registered speakers: {', '.join(speakers)}")
        
    except Exception as e:
        console.print(f"💥 [red]Error loading session: {str(e)}[/red]")


def main():
    """Main CLI entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\\n🛑 Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        console.print(f"\\n💥 [red]Unexpected error: {str(e)}[/red]")
        sys.exit(1)


if __name__ == '__main__':
    main()