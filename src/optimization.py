"""
Performance Optimization and Error Handling Utilities

This module provides utilities for optimizing performance and handling
errors gracefully in the speech translation system.
"""

import logging
import time
import psutil
import torch
from typing import Dict, Any, Optional, Callable
from functools import wraps
from pathlib import Path
import json

from ..config import SAMPLE_RATE


class PerformanceMonitor:
    """Monitor system performance and resource usage."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'processing_times': [],
            'model_load_times': {}
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system information."""
        info = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'disk_usage_percent': psutil.disk_usage('/').percent if hasattr(psutil.disk_usage, '__call__') else 0,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            try:
                info['gpu_memory_allocated'] = torch.cuda.memory_allocated() / (1024**3)  # GB
                info['gpu_memory_reserved'] = torch.cuda.memory_reserved() / (1024**3)   # GB
            except:
                info['gpu_memory_allocated'] = 0
                info['gpu_memory_reserved'] = 0
        
        return info
    
    def log_system_status(self):
        """Log current system status."""
        info = self.get_system_info()
        self.logger.info(f"System Status - CPU: {info['cpu_percent']:.1f}%, "
                        f"Memory: {info['memory_percent']:.1f}%, "
                        f"Available Memory: {info['available_memory_gb']:.1f}GB")
        
        if info['cuda_available']:
            self.logger.info(f"GPU Memory - Allocated: {info['gpu_memory_allocated']:.2f}GB, "
                           f"Reserved: {info['gpu_memory_reserved']:.2f}GB")
    
    def record_processing_time(self, operation: str, duration: float):
        """Record processing time for an operation."""
        self.metrics['processing_times'].append({
            'operation': operation,
            'duration': duration,
            'timestamp': time.time()
        })
        
        self.logger.debug(f"Operation '{operation}' completed in {duration:.2f}s")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        processing_times = self.metrics['processing_times']
        
        if not processing_times:
            return {'message': 'No performance data available'}
        
        # Group by operation
        operations = {}
        for entry in processing_times:
            op = entry['operation']
            if op not in operations:
                operations[op] = []
            operations[op].append(entry['duration'])
        
        # Calculate statistics
        summary = {}
        for op, times in operations.items():
            summary[op] = {
                'count': len(times),
                'total_time': sum(times),
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times)
            }
        
        return summary


def performance_monitor(operation_name: Optional[str] = None):
    """Decorator to monitor function performance."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log performance
                op_name = operation_name or func.__name__
                logging.getLogger(__name__).debug(f"{op_name} completed in {duration:.2f}s")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logging.getLogger(__name__).error(f"{func.__name__} failed after {duration:.2f}s: {str(e)}")
                raise
        
        return wrapper
    return decorator


class MemoryManager:
    """Manage memory usage and cleanup."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory."""
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self.logger.debug("GPU memory cleared")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup GPU memory: {str(e)}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory_info = {
            'system_memory_percent': psutil.virtual_memory().percent,
            'system_memory_available_gb': psutil.virtual_memory().available / (1024**3)
        }
        
        if torch.cuda.is_available():
            try:
                memory_info['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
                memory_info['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
            except:
                memory_info['gpu_memory_allocated_gb'] = 0
                memory_info['gpu_memory_reserved_gb'] = 0
        
        return memory_info
    
    def check_memory_threshold(self, threshold_percent: float = 85.0) -> bool:
        """Check if memory usage exceeds threshold."""
        usage = self.get_memory_usage()
        
        if usage['system_memory_percent'] > threshold_percent:
            self.logger.warning(f"High system memory usage: {usage['system_memory_percent']:.1f}%")
            return True
        
        return False
    
    def optimize_memory_usage(self):
        """Optimize memory usage."""
        self.cleanup_gpu_memory()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        self.logger.debug("Memory optimization completed")


class ErrorHandler:
    """Enhanced error handling with recovery strategies."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_counts = {}
        self.recovery_strategies = {}
    
    def register_recovery_strategy(self, error_type: type, strategy: Callable):
        """Register a recovery strategy for specific error type."""
        self.recovery_strategies[error_type] = strategy
    
    def handle_error(self, error: Exception, context: str = "") -> bool:
        """
        Handle error with recovery strategy.
        
        Returns:
            bool: True if recovered, False if not
        """
        error_type = type(error)
        error_key = f"{error_type.__name__}_{context}"
        
        # Track error frequency
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        self.logger.error(f"Error in {context}: {str(error)} (count: {self.error_counts[error_key]})")
        
        # Try recovery strategy
        if error_type in self.recovery_strategies:
            try:
                self.logger.info(f"Attempting recovery for {error_type.__name__}")
                self.recovery_strategies[error_type](error)
                return True
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed: {str(recovery_error)}")
        
        return False
    
    def get_error_statistics(self) -> Dict[str, int]:
        """Get error statistics."""
        return self.error_counts.copy()


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, exponential_backoff: bool = True):
    """Decorator to retry function on failure."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        wait_time = delay * (2 ** attempt if exponential_backoff else 1)
                        logging.getLogger(__name__).warning(
                            f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logging.getLogger(__name__).error(f"All {max_retries + 1} attempts failed")
            
            raise last_exception
        
        return wrapper
    return decorator


class ModelOptimizer:
    """Optimize model performance and resource usage."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.optimization_cache = {}
    
    def optimize_for_device(self, device: str) -> Dict[str, Any]:
        """Get optimization settings for specific device."""
        optimizations = {
            'cpu': {
                'torch_threads': min(4, torch.get_num_threads()),
                'batch_size': 1,
                'precision': 'float32',
                'num_workers': 0
            },
            'cuda': {
                'torch_threads': torch.get_num_threads(),
                'batch_size': 4,
                'precision': 'float16',
                'num_workers': 2
            }
        }
        
        return optimizations.get(device, optimizations['cpu'])
    
    def optimize_audio_processing(self, audio_length: float, device: str) -> Dict[str, Any]:
        """Optimize audio processing parameters based on audio length and device."""
        settings = {
            'chunk_size': 30.0,  # seconds
            'overlap': 0.1,      # 10% overlap
            'sample_rate': SAMPLE_RATE
        }
        
        # Adjust chunk size based on audio length and device capabilities
        if device == 'cuda':
            # GPU can handle larger chunks
            settings['chunk_size'] = min(60.0, audio_length)
        else:
            # CPU: smaller chunks for better performance
            settings['chunk_size'] = min(30.0, audio_length)
        
        # For very short audio, process as single chunk
        if audio_length < 10.0:
            settings['chunk_size'] = audio_length
            settings['overlap'] = 0.0
        
        return settings
    
    def get_recommended_model_sizes(self, device: str, available_memory_gb: float) -> Dict[str, str]:
        """Get recommended model sizes based on available resources."""
        recommendations = {}
        
        if device == 'cpu':
            # CPU recommendations based on memory
            if available_memory_gb >= 16:
                recommendations = {
                    'whisper': 'base',
                    'translation': 'local',
                    'tts': 'tts_models/multilingual/multi-dataset/xtts_v2'
                }
            elif available_memory_gb >= 8:
                recommendations = {
                    'whisper': 'tiny',
                    'translation': 'google',
                    'tts': 'tts_models/en/ljspeech/tacotron2-DDC'
                }
            else:
                recommendations = {
                    'whisper': 'tiny',
                    'translation': 'google',
                    'tts': 'tts_models/en/ljspeech/speedy_speech'
                }
        
        else:  # GPU
            # GPU recommendations
            if available_memory_gb >= 12:
                recommendations = {
                    'whisper': 'large',
                    'translation': 'local',
                    'tts': 'tts_models/multilingual/multi-dataset/xtts_v2'
                }
            elif available_memory_gb >= 6:
                recommendations = {
                    'whisper': 'medium',
                    'translation': 'local',
                    'tts': 'tts_models/multilingual/multi-dataset/xtts_v2'
                }
            else:
                recommendations = {
                    'whisper': 'base',
                    'translation': 'google',
                    'tts': 'tts_models/en/ljspeech/tacotron2-DDC'
                }
        
        return recommendations


class ConfigurationOptimizer:
    """Optimize system configuration based on hardware and usage patterns."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.performance_monitor = PerformanceMonitor()
        self.memory_manager = MemoryManager()
        self.model_optimizer = ModelOptimizer()
    
    def analyze_system(self) -> Dict[str, Any]:
        """Analyze current system capabilities."""
        system_info = self.performance_monitor.get_system_info()
        memory_info = self.memory_manager.get_memory_usage()
        
        analysis = {
            'system_info': system_info,
            'memory_info': memory_info,
            'recommended_device': 'cuda' if system_info['cuda_available'] else 'cpu',
            'performance_level': 'high' if system_info['cuda_available'] and memory_info['system_memory_available_gb'] > 12 else 'standard'
        }
        
        # Model recommendations
        device = analysis['recommended_device']
        available_memory = memory_info['system_memory_available_gb']
        
        analysis['recommended_models'] = self.model_optimizer.get_recommended_model_sizes(
            device, available_memory
        )
        
        return analysis
    
    def generate_optimal_config(self, usage_pattern: str = 'general') -> Dict[str, Any]:
        """
        Generate optimal configuration based on system analysis.
        
        Args:
            usage_pattern: 'realtime', 'batch', 'quality', or 'general'
        """
        analysis = self.analyze_system()
        
        base_config = {
            'device': analysis['recommended_device'],
            'speech_model': analysis['recommended_models']['whisper'],
            'translation_engine': analysis['recommended_models']['translation'],
            'tts_model': analysis['recommended_models']['tts']
        }
        
        # Adjust based on usage pattern
        if usage_pattern == 'realtime':
            # Optimize for speed
            base_config.update({
                'speech_model': 'tiny',
                'translation_engine': 'google',  # Faster API calls
                'audio_chunk_size': 15.0,  # Smaller chunks for faster processing
                'enable_caching': True
            })
        
        elif usage_pattern == 'batch':
            # Optimize for throughput
            base_config.update({
                'audio_chunk_size': 60.0,  # Larger chunks for batch processing
                'batch_size': 8,
                'enable_parallel_processing': True
            })
        
        elif usage_pattern == 'quality':
            # Optimize for quality
            if analysis['system_info']['cuda_available']:
                base_config.update({
                    'speech_model': 'large',
                    'translation_engine': 'local',
                    'voice_sample_requirements': {
                        'min_duration': 30.0,
                        'min_samples': 5
                    }
                })
        
        return base_config
    
    def save_config(self, config: Dict[str, Any], config_path: str):
        """Save configuration to file."""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Configuration saved to: {config_file}")
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        config_file = Path(config_path)
        
        if not config_file.exists():
            self.logger.warning(f"Configuration file not found: {config_file}")
            return self.generate_optimal_config()
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        self.logger.info(f"Configuration loaded from: {config_file}")
        return config


# Utility functions for common optimizations
def optimize_torch_settings(device: str):
    """Optimize PyTorch settings for the given device."""
    if device == 'cpu':
        # Optimize for CPU
        torch.set_num_threads(min(4, torch.get_num_threads()))
        torch.set_num_interop_threads(2)
    else:
        # GPU optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def setup_error_recovery():
    """Setup common error recovery strategies."""
    error_handler = ErrorHandler()
    memory_manager = MemoryManager()
    
    # GPU out of memory recovery
    def gpu_memory_recovery(error):
        memory_manager.cleanup_gpu_memory()
        time.sleep(1)  # Wait for cleanup
    
    # Network error recovery for translation
    def network_recovery(error):
        time.sleep(2)  # Wait before retry
    
    error_handler.register_recovery_strategy(RuntimeError, gpu_memory_recovery)
    error_handler.register_recovery_strategy(ConnectionError, network_recovery)
    
    return error_handler


# Performance profiling decorator
def profile_performance(func):
    """Decorator to profile function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        import cProfile
        import pstats
        import io
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
            
            # Print performance stats
            s = io.StringIO()
            stats = pstats.Stats(profiler, stream=s)
            stats.sort_stats('cumulative')
            stats.print_stats(10)  # Top 10 functions
            
            logging.getLogger(__name__).debug(f"Performance profile for {func.__name__}:\\n{s.getvalue()}")
        
        return result
    
    return wrapper