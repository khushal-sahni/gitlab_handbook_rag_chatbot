"""
Centralized logging configuration for the GitLab RAG Chatbot.

This module provides comprehensive logging setup with file and console handlers,
structured formatting, and appropriate log levels for different components.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from ..config.settings import settings


def setup_application_logging(
    log_level: str = "INFO",
    log_file_path: Optional[Path] = None,
    enable_console_logging: bool = True,
    enable_file_logging: bool = True,
    max_log_file_size_mb: int = 10,
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up comprehensive logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file_path: Path to log file (uses config default if None)
        enable_console_logging: Whether to log to console
        enable_file_logging: Whether to log to file
        max_log_file_size_mb: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
        
    Returns:
        Configured logger instance
    """
    # Convert log level string to logging constant
    numeric_log_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_log_level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if enable_console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if enable_file_logging:
        log_file = log_file_path or settings.ingestion_log_path
        
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_log_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Configure specific loggers for third-party libraries
    _configure_third_party_loggers()
    
    # Log configuration summary
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {log_level}, Console: {enable_console_logging}, File: {enable_file_logging}")
    
    return root_logger


def _configure_third_party_loggers():
    """Configure logging levels for third-party libraries to reduce noise."""
    
    # Reduce verbosity of common third-party libraries
    third_party_loggers = {
        'requests': logging.WARNING,
        'urllib3': logging.WARNING,
        'chromadb': logging.WARNING,
        'openai': logging.WARNING,
        'google': logging.WARNING,
        'httpx': logging.WARNING,
        'httpcore': logging.WARNING
    }
    
    for logger_name, level in third_party_loggers.items():
        logging.getLogger(logger_name).setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__ from calling module)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggingContext:
    """Context manager for temporary logging level changes."""
    
    def __init__(self, logger_name: str, temporary_level: str):
        """
        Initialize logging context.
        
        Args:
            logger_name: Name of logger to modify
            temporary_level: Temporary logging level to set
        """
        self.logger = logging.getLogger(logger_name)
        self.temporary_level = getattr(logging, temporary_level.upper())
        self.original_level = self.logger.level
    
    def __enter__(self):
        """Set temporary logging level."""
        self.logger.setLevel(self.temporary_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original logging level."""
        self.logger.setLevel(self.original_level)
