"""
Core functionality for GitLab RAG Chatbot.

This package contains the main business logic components:
- Document ingestion and processing
- Vector storage and retrieval 
- Embedding generation
- Progress tracking and checkpointing
"""

from .retrieval.vector_store import DocumentRetriever
from .embeddings.providers import EmbeddingProvider
from .ingestion.progress_tracker import IngestionProgressTracker

__all__ = [
    "DocumentRetriever",
    "EmbeddingProvider", 
    "IngestionProgressTracker"
]
