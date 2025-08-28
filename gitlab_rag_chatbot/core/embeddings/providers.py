"""
Embedding generation providers for different AI services.

This module provides a unified interface for generating text embeddings
using various AI providers (Gemini, OpenAI) with robust error handling
and retry logic.
"""

import time
import logging
from typing import List, Callable, Protocol
from abc import ABC, abstractmethod

from ...config.settings import settings
from .retry_handler import with_retry

logger = logging.getLogger(__name__)


class EmbeddingFunction(Protocol):
    """Protocol defining the interface for embedding functions."""
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        ...


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def create_embedding_function(self) -> EmbeddingFunction:
        """Create and return an embedding function for this provider."""
        pass
    
    @abstractmethod
    def validate_configuration(self) -> None:
        """Validate that the provider is properly configured."""
        pass


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """
    Embedding provider using Google's Gemini API.
    
    Provides text embeddings using Google's text-embedding-004 model
    with built-in retry logic and rate limiting.
    """
    
    def __init__(self):
        """Initialize Gemini embedding provider."""
        self.api_key = settings.gemini_api_key
        self.validate_configuration()
        
        # Lazy import to avoid dependency issues
        try:
            import google.generativeai as genai
            self.genai = genai
            genai.configure(api_key=self.api_key)
            logger.info("Initialized Gemini embedding provider")
        except ImportError as e:
            raise RuntimeError("google-generativeai package not installed") from e
    
    def validate_configuration(self) -> None:
        """Validate Gemini API configuration."""
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
    
    @with_retry()
    def _generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text with retry logic.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            result = self.genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
            
        except Exception as e:
            logger.error(f"Gemini embedding generation failed: {e}")
            raise
    
    def create_embedding_function(self) -> EmbeddingFunction:
        """
        Create embedding function for batch processing.
        
        Returns:
            Function that takes list of texts and returns list of embeddings
        """
        def embed_texts(texts: List[str]) -> List[List[float]]:
            """
            Generate embeddings for multiple texts with rate limiting.
            
            Args:
                texts: List of texts to embed
                
            Returns:
                List of embedding vectors
            """
            embeddings = []
            total_texts = len(texts)
            
            logger.debug(f"Generating embeddings for {total_texts} texts using Gemini")
            
            for i, text in enumerate(texts):
                logger.debug(f"Processing text {i+1}/{total_texts}")
                
                embedding = self._generate_single_embedding(text)
                embeddings.append(embedding)
                
                # Rate limiting: small delay between requests
                if i < total_texts - 1:
                    time.sleep(0.1)
            
            logger.info(f"Successfully generated {len(embeddings)} embeddings using Gemini")
            return embeddings
        
        return embed_texts


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    Embedding provider using OpenAI's API.
    
    Provides text embeddings using OpenAI's text-embedding-3-small model
    with built-in retry logic and batch processing support.
    """
    
    def __init__(self):
        """Initialize OpenAI embedding provider."""
        self.api_key = settings.openai_api_key
        self.validate_configuration()
        
        # Lazy import to avoid dependency issues
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            logger.info("Initialized OpenAI embedding provider")
        except ImportError as e:
            raise RuntimeError("openai package not installed") from e
    
    def validate_configuration(self) -> None:
        """Validate OpenAI API configuration."""
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
    
    def create_embedding_function(self) -> EmbeddingFunction:
        """
        Create embedding function for batch processing.
        
        Returns:
            Function that takes list of texts and returns list of embeddings
        """
        @with_retry()
        def embed_texts(texts: List[str]) -> List[List[float]]:
            """
            Generate embeddings for multiple texts using OpenAI API.
            
            Args:
                texts: List of texts to embed
                
            Returns:
                List of embedding vectors
            """
            logger.debug(f"Generating embeddings for {len(texts)} texts using OpenAI")
            
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texts,
                    timeout=settings.api_timeout_seconds
                )
                
                embeddings = [embedding_data.embedding for embedding_data in response.data]
                logger.info(f"Successfully generated {len(embeddings)} embeddings using OpenAI")
                return embeddings
                
            except Exception as e:
                logger.error(f"OpenAI embedding generation failed: {e}")
                raise
        
        return embed_texts


class EmbeddingProvider:
    """
    Factory class for creating embedding providers.
    
    Provides a unified interface for accessing different embedding providers
    based on configuration settings.
    """
    
    _PROVIDERS = {
        "gemini": GeminiEmbeddingProvider,
        "openai": OpenAIEmbeddingProvider
    }
    
    @classmethod
    def create_embedding_function(cls, provider_name: str = None) -> EmbeddingFunction:
        """
        Create an embedding function for the specified provider.
        
        Args:
            provider_name: Name of the provider ("gemini" or "openai").
                          Uses config default if None.
                          
        Returns:
            Embedding function that can generate embeddings for text lists
            
        Raises:
            ValueError: If provider is not supported
            RuntimeError: If provider configuration is invalid
        """
        if provider_name is None:
            provider_name = settings.embedding_provider
        
        provider_name = provider_name.lower()
        
        if provider_name not in cls._PROVIDERS:
            supported_providers = ", ".join(cls._PROVIDERS.keys())
            raise ValueError(
                f"Unsupported embedding provider: {provider_name}. "
                f"Supported providers: {supported_providers}"
            )
        
        try:
            provider_class = cls._PROVIDERS[provider_name]
            provider_instance = provider_class()
            return provider_instance.create_embedding_function()
            
        except Exception as e:
            logger.error(f"Failed to initialize {provider_name} embedding provider: {e}")
            raise RuntimeError(f"Embedding provider initialization failed: {e}") from e
    
    @classmethod
    def get_supported_providers(cls) -> List[str]:
        """
        Get list of supported embedding providers.
        
        Returns:
            List of provider names
        """
        return list(cls._PROVIDERS.keys())


# Convenience function for backward compatibility
def get_embedding_function(provider_name: str = None) -> EmbeddingFunction:
    """
    Get an embedding function for the specified provider.
    
    This is a convenience function that wraps EmbeddingProvider.create_embedding_function()
    for backward compatibility with existing code.
    
    Args:
        provider_name: Name of the provider to use
        
    Returns:
        Embedding function
    """
    return EmbeddingProvider.create_embedding_function(provider_name)
