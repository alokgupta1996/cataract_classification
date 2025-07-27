"""
Centralized logging configuration for the JIVI Cataract Classification project.
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    console_output: bool = True
) -> logging.Logger:
    """
    Setup centralized logging configuration.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (str, optional): Path to log file
        log_format (str): Log message format
        console_output (bool): Whether to output to console
    
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Convert string level to logging level
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        # Default log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_log_file = logs_dir / f"jivi_cataract_{timestamp}.log"
        file_handler = logging.FileHandler(default_log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name (str): Logger name (usually __name__)
    
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)


def log_system_info(logger: logging.Logger):
    """
    Log system information for debugging.
    
    Args:
        logger (logging.Logger): Logger instance
    """
    import platform
    import torch
    
    logger.info("=" * 60)
    logger.info("🚀 JIVI Cataract Classification System")
    logger.info("=" * 60)
    logger.info(f"📱 Platform: {platform.platform()}")
    logger.info(f"🐍 Python Version: {sys.version}")
    logger.info(f"🔥 PyTorch Version: {torch.__version__}")
    logger.info(f"⚡ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"🎮 CUDA Version: {torch.version.cuda}")
        logger.info(f"🎮 GPU Count: {torch.cuda.device_count()}")
        logger.info(f"🎮 GPU Name: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 60)


def log_model_info(logger: logging.Logger, model_name: str, model_path: str, device: str):
    """
    Log model loading information.
    
    Args:
        logger (logging.Logger): Logger instance
        model_name (str): Name of the model
        model_path (str): Path to the model file
        device (str): Device the model is loaded on
    """
    logger.info(f"📦 Loading {model_name} model...")
    logger.info(f"📁 Model path: {model_path}")
    logger.info(f"⚡ Device: {device}")
    
    # Check if model file exists
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        logger.info(f"📏 Model file size: {file_size:.2f} MB")
    else:
        logger.warning(f"⚠️ Model file not found: {model_path}")


def log_performance_metrics(logger: logging.Logger, metrics: dict, model_name: str):
    """
    Log performance metrics.
    
    Args:
        logger (logging.Logger): Logger instance
        metrics (dict): Dictionary of performance metrics
        model_name (str): Name of the model
    """
    logger.info(f"📊 {model_name} Performance Metrics:")
    logger.info("-" * 40)
    
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric_name}: {value:.4f}")
        else:
            logger.info(f"  {metric_name}: {value}")
    
    logger.info("-" * 40)


def log_api_request(logger: logging.Logger, endpoint: str, method: str, processing_time: float, status_code: int):
    """
    Log API request information.
    
    Args:
        logger (logging.Logger): Logger instance
        endpoint (str): API endpoint
        method (str): HTTP method
        processing_time (float): Request processing time
        status_code (int): HTTP status code
    """
    status_emoji = "✅" if status_code == 200 else "❌"
    logger.info(f"{status_emoji} {method} {endpoint} - {status_code} ({processing_time:.3f}s)")


def log_error_with_context(logger: logging.Logger, error: Exception, context: str = ""):
    """
    Log error with context information.
    
    Args:
        logger (logging.Logger): Logger instance
        error (Exception): The error that occurred
        context (str): Additional context information
    """
    logger.error(f"💥 Error in {context}: {str(error)}")
    logger.error(f"🔍 Error type: {type(error).__name__}")
    import traceback
    logger.error(f"📋 Traceback: {traceback.format_exc()}")


def log_memory_usage(logger: logging.Logger):
    """
    Log current memory usage.
    
    Args:
        logger (logging.Logger): Logger instance
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        logger.info(f"💾 Memory usage: {memory_mb:.2f} MB")
    except ImportError:
        logger.debug("psutil not available for memory monitoring")


def log_gpu_memory(logger: logging.Logger):
    """
    Log GPU memory usage if CUDA is available.
    
    Args:
        logger (logging.Logger): Logger instance
    """
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
                memory_reserved = torch.cuda.memory_reserved(i) / 1024 / 1024
                logger.info(f"🎮 GPU {i} memory - Allocated: {memory_allocated:.2f} MB, Reserved: {memory_reserved:.2f} MB")
    except Exception as e:
        logger.debug(f"Could not log GPU memory: {e}")


# Convenience function for quick setup
def setup_project_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Quick setup for project logging.
    
    Args:
        log_level (str): Logging level
    
    Returns:
        logging.Logger: Configured logger
    """
    return setup_logging(log_level=log_level) 