"""
Application settings management with environment variable support.

This module provides centralized configuration management with type safety,
validation, and comprehensive documentation for all settings.
"""

import os
from pathlib import Path
from typing import Optional, Set, List
from dotenv import load_dotenv

from .constants import (
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_TOP_K_RESULTS,
    DEFAULT_MIN_SIMILARITY_SCORE, DEFAULT_MAX_PAGES_TO_CRAWL,
    DEFAULT_EMBEDDING_BATCH_SIZE, DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_CRAWL_DELAY_SECONDS, DEFAULT_API_TIMEOUT_SECONDS,
    DEFAULT_MAX_API_RETRIES, DEFAULT_RETRY_BASE_DELAY_SECONDS,
    DEFAULT_RETRY_MAX_DELAY_SECONDS, DEFAULT_CHECKPOINT_INTERVAL,
    DEFAULT_RESUME_FROM_CHECKPOINT, SUPPORTED_EMBEDDING_PROVIDERS,
    GITLAB_HANDBOOK_DOMAINS, GITLAB_SEED_URLS, DEFAULT_USER_AGENT,
    DATA_DIRECTORY_NAME, CACHE_DIRECTORY_NAME, VECTOR_STORE_DIRECTORY_NAME,
    LOGS_DIRECTORY_NAME, FEEDBACK_CSV_FILENAME, INGESTION_LOG_FILENAME,
    CHECKPOINT_FILENAME
)


class ApplicationSettings:
    """
    Centralized application settings with environment variable support.
    
    All configuration values are loaded from environment variables with
    sensible defaults. This class provides type safety and validation
    for all application settings.
    """
    
    def __init__(self, env_file_path: Optional[str] = None):
        """
        Initialize settings by loading environment variables.
        
        Args:
            env_file_path: Optional path to .env file. If None, uses default .env
        """
        # Load environment variables
        if env_file_path:
            load_dotenv(env_file_path)
        else:
            load_dotenv()
        
        # Initialize all settings
        self._load_api_settings()
        self._load_document_processing_settings()
        self._load_retrieval_settings()
        self._load_ingestion_performance_settings()
        self._load_api_reliability_settings()
        self._load_persistence_settings()
        self._load_directory_settings()
        self._validate_settings()
    
    def _load_api_settings(self) -> None:
        """Load API provider settings."""
        self.embedding_provider = os.getenv("PROVIDER", "gemini").lower()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
    
    def _load_document_processing_settings(self) -> None:
        """Load document processing configuration."""
        self.chunk_size = int(os.getenv("CHUNK_SIZE", DEFAULT_CHUNK_SIZE))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP))
    
    def _load_retrieval_settings(self) -> None:
        """Load retrieval and search configuration."""
        self.top_k_results = int(os.getenv("TOP_K", DEFAULT_TOP_K_RESULTS))
        self.minimum_similarity_score = float(os.getenv("MIN_SCORE", DEFAULT_MIN_SIMILARITY_SCORE))
    
    def _load_ingestion_performance_settings(self) -> None:
        """Load ingestion performance optimization settings."""
        self.max_pages_to_crawl = int(os.getenv("MAX_PAGES", DEFAULT_MAX_PAGES_TO_CRAWL))
        self.embedding_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE))
        self.crawl_delay_seconds = float(os.getenv("CRAWL_DELAY", DEFAULT_CRAWL_DELAY_SECONDS))
        self.request_timeout_seconds = int(os.getenv("REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT_SECONDS))
    
    def _load_api_reliability_settings(self) -> None:
        """Load API retry and reliability configuration."""
        self.max_api_retries = int(os.getenv("MAX_RETRIES", DEFAULT_MAX_API_RETRIES))
        self.retry_base_delay_seconds = float(os.getenv("RETRY_BASE_DELAY", DEFAULT_RETRY_BASE_DELAY_SECONDS))
        self.retry_max_delay_seconds = float(os.getenv("RETRY_MAX_DELAY", DEFAULT_RETRY_MAX_DELAY_SECONDS))
        self.api_timeout_seconds = int(os.getenv("API_TIMEOUT", DEFAULT_API_TIMEOUT_SECONDS))
    
    def _load_persistence_settings(self) -> None:
        """Load checkpoint and persistence configuration."""
        self.checkpoint_interval = int(os.getenv("CHECKPOINT_INTERVAL", DEFAULT_CHECKPOINT_INTERVAL))
        self.resume_from_checkpoint = os.getenv("RESUME_FROM_CHECKPOINT", str(DEFAULT_RESUME_FROM_CHECKPOINT)).lower() == "true"
    
    def _load_directory_settings(self) -> None:
        """Load directory and file path settings."""
        # Base project directory (parent of this config module)
        self.project_root = Path(__file__).parent.parent.parent
        
        # Data directories
        self.data_directory = self.project_root / "src" / DATA_DIRECTORY_NAME
        self.cache_directory = self.data_directory / CACHE_DIRECTORY_NAME
        self.vector_store_directory = self.data_directory / VECTOR_STORE_DIRECTORY_NAME
        self.logs_directory = self.project_root / "src" / LOGS_DIRECTORY_NAME
        
        # File paths
        self.feedback_csv_path = self.logs_directory / FEEDBACK_CSV_FILENAME
        self.ingestion_log_path = self.logs_directory / INGESTION_LOG_FILENAME
        self.checkpoint_file_path = self.data_directory / CHECKPOINT_FILENAME
        
        # Web crawling settings
        self.seed_urls = GITLAB_SEED_URLS
        self.allowed_domains = GITLAB_HANDBOOK_DOMAINS
        self.user_agent = DEFAULT_USER_AGENT
    
    def _validate_settings(self) -> None:
        """Validate configuration settings."""
        # Validate embedding provider
        if self.embedding_provider not in SUPPORTED_EMBEDDING_PROVIDERS:
            raise ValueError(
                f"Unsupported embedding provider: {self.embedding_provider}. "
                f"Supported providers: {', '.join(SUPPORTED_EMBEDDING_PROVIDERS)}"
            )
        
        # Validate API keys
        if self.embedding_provider == "gemini" and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required when using Gemini provider")
        
        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required when using OpenAI provider")
        
        # Validate numeric ranges
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be non-negative and less than CHUNK_SIZE")
        
        if self.top_k_results <= 0:
            raise ValueError("TOP_K must be positive")
        
        if not 0 <= self.minimum_similarity_score <= 1:
            raise ValueError("MIN_SCORE must be between 0 and 1")
        
        if self.max_pages_to_crawl <= 0:
            raise ValueError("MAX_PAGES must be positive")
        
        if self.embedding_batch_size <= 0:
            raise ValueError("EMBEDDING_BATCH_SIZE must be positive")
    
    def ensure_directories_exist(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.data_directory,
            self.cache_directory,
            self.vector_store_directory,
            self.logs_directory
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_configuration_summary(self) -> dict:
        """
        Get a summary of current configuration for logging/debugging.
        
        Returns:
            Dictionary containing non-sensitive configuration values
        """
        return {
            "embedding_provider": self.embedding_provider,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k_results": self.top_k_results,
            "minimum_similarity_score": self.minimum_similarity_score,
            "max_pages_to_crawl": self.max_pages_to_crawl,
            "embedding_batch_size": self.embedding_batch_size,
            "crawl_delay_seconds": self.crawl_delay_seconds,
            "request_timeout_seconds": self.request_timeout_seconds,
            "max_api_retries": self.max_api_retries,
            "checkpoint_interval": self.checkpoint_interval,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "data_directory": str(self.data_directory),
            "vector_store_directory": str(self.vector_store_directory)
        }


# Global settings instance
settings = ApplicationSettings()
