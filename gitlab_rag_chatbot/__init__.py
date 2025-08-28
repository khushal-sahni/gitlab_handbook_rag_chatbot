"""
GitLab RAG Chatbot - Production-grade RAG system for GitLab Handbook and Direction docs.

This package provides a complete RAG (Retrieval-Augmented Generation) solution
for querying GitLab's public documentation with semantic search and AI-powered responses.
"""

__version__ = "1.0.0"
__author__ = "GitLab RAG Team"
__description__ = "Production-grade RAG chatbot for GitLab documentation"

from .core.retrieval.vector_store import DocumentRetriever
from .core.embeddings.providers import EmbeddingProvider

__all__ = [
    "DocumentRetriever", 
    "EmbeddingProvider"
]
