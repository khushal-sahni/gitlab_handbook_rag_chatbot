"""Configuration management for GitLab RAG Chatbot."""

from .settings import ApplicationSettings
from .constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K_RESULTS,
    DEFAULT_MIN_SIMILARITY_SCORE,
    SUPPORTED_EMBEDDING_PROVIDERS,
    GITLAB_HANDBOOK_DOMAINS
)

__all__ = [
    "ApplicationSettings",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP", 
    "DEFAULT_TOP_K_RESULTS",
    "DEFAULT_MIN_SIMILARITY_SCORE",
    "SUPPORTED_EMBEDDING_PROVIDERS",
    "GITLAB_HANDBOOK_DOMAINS"
]
