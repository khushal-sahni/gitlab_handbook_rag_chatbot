"""
Robust API retry handling with exponential backoff.

This module provides decorators and utilities for handling API failures
with intelligent retry logic, rate limiting, and error classification.
"""

import time
import logging
from typing import Callable, Any, List, Set
from functools import wraps

from ...config.settings import settings

logger = logging.getLogger(__name__)


class APIRetryHandler:
    """
    Handles API retry logic with exponential backoff and intelligent error classification.
    
    This class provides robust retry mechanisms for API calls that may fail due to
    network issues, rate limiting, or temporary service unavailability.
    """
    
    # Error keywords that indicate retryable failures
    RETRYABLE_ERROR_KEYWORDS: Set[str] = {
        'timeout', 'deadline exceeded', '504', '503', '502', '500',
        'rate limit', 'quota', 'temporarily unavailable', 'service unavailable',
        'connection error', 'network error', 'too many requests'
    }
    
    def __init__(self, 
                 max_retries: int = None,
                 base_delay: float = None,
                 max_delay: float = None):
        """
        Initialize retry handler with configuration.
        
        Args:
            max_retries: Maximum number of retry attempts (uses config default if None)
            base_delay: Base delay in seconds for exponential backoff (uses config default if None)
            max_delay: Maximum delay in seconds (uses config default if None)
        """
        self.max_retries = max_retries or settings.max_api_retries
        self.base_delay = base_delay or settings.retry_base_delay_seconds
        self.max_delay = max_delay or settings.retry_max_delay_seconds
    
    def is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable based on error message analysis.
        
        Args:
            error: The exception to analyze
            
        Returns:
            True if the error appears to be retryable, False otherwise
        """
        error_message = str(error).lower()
        return any(keyword in error_message for keyword in self.RETRYABLE_ERROR_KEYWORDS)
    
    def calculate_delay(self, attempt_number: int) -> float:
        """
        Calculate delay for exponential backoff with jitter.
        
        Args:
            attempt_number: Current attempt number (0-based)
            
        Returns:
            Delay in seconds before next retry
        """
        exponential_delay = self.base_delay * (2 ** attempt_number)
        return min(exponential_delay, self.max_delay)
    
    def retry_with_backoff(self, func: Callable) -> Callable:
        """
        Decorator that adds retry logic with exponential backoff to a function.
        
        Args:
            func: Function to wrap with retry logic
            
        Returns:
            Wrapped function with retry capabilities
        """
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    last_exception = e
                    
                    # Check if error is retryable
                    if not self.is_retryable_error(e):
                        logger.error(f"Non-retryable error in {func.__name__}: {e}")
                        raise e
                    
                    # If this was the last attempt, raise the exception
                    if attempt >= self.max_retries:
                        break
                    
                    # Calculate delay and log retry attempt
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        f"API error in {func.__name__} (attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                    )
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    
                    time.sleep(delay)
            
            # All retries exhausted
            logger.error(
                f"All {self.max_retries + 1} attempts failed for {func.__name__}. "
                f"Last error: {last_exception}"
            )
            raise last_exception
        
        return wrapper


# Global retry handler instance
default_retry_handler = APIRetryHandler()


def with_retry(max_retries: int = None, 
               base_delay: float = None, 
               max_delay: float = None) -> Callable:
    """
    Convenience decorator for adding retry logic to functions.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay for exponential backoff
        max_delay: Maximum delay between retries
        
    Returns:
        Decorator function
        
    Example:
        @with_retry(max_retries=3, base_delay=1.0)
        def api_call():
            # Your API call here
            pass
    """
    retry_handler = APIRetryHandler(max_retries, base_delay, max_delay)
    return retry_handler.retry_with_backoff
