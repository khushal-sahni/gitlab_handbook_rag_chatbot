"""
Vector storage and document retrieval using ChromaDB.

This module provides a high-level interface for storing document embeddings
and performing semantic similarity searches using ChromaDB as the backend.
"""

import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from ...config.settings import settings

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """
    High-level interface for document storage and retrieval using vector embeddings.
    
    This class manages interactions with ChromaDB for storing document chunks
    and their embeddings, and performing semantic similarity searches.
    """
    
    def __init__(self, collection_name: str = "gitlab_documentation"):
        """
        Initialize the document retriever with ChromaDB backend.
        
        Args:
            collection_name: Name of the ChromaDB collection to use
        """
        self.collection_name = collection_name
        self._initialize_vector_store()
        logger.info(f"Initialized DocumentRetriever with collection: {collection_name}")
    
    def _initialize_vector_store(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            # Ensure vector store directory exists
            settings.ensure_directories_exist()
            
            # Use PersistentClient for ChromaDB 0.5.x compatibility
            self.chroma_client = chromadb.PersistentClient(
                path=str(settings.vector_store_directory)
            )
            
            # Get or create collection
            self.document_collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name
            )
            
            logger.info(f"ChromaDB initialized at: {settings.vector_store_directory}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise RuntimeError(f"Vector store initialization failed: {e}") from e
    
    def add_documents(self, 
                     document_ids: List[str],
                     document_texts: List[str], 
                     document_metadata: List[Dict[str, Any]],
                     document_embeddings: List[List[float]]) -> None:
        """
        Add documents with their embeddings to the vector store.
        
        Args:
            document_ids: Unique identifiers for each document chunk
            document_texts: The actual text content of document chunks
            document_metadata: Metadata associated with each document (e.g., source URL)
            document_embeddings: Vector embeddings for each document chunk
            
        Raises:
            ValueError: If input lists have different lengths
            RuntimeError: If ChromaDB operation fails
        """
        # Validate input consistency
        input_lengths = [
            len(document_ids), len(document_texts), 
            len(document_metadata), len(document_embeddings)
        ]
        
        if len(set(input_lengths)) != 1:
            raise ValueError(
                f"All input lists must have the same length. "
                f"Got lengths: {input_lengths}"
            )
        
        batch_size = len(document_ids)
        
        try:
            self.document_collection.add(
                ids=document_ids,
                documents=document_texts,
                metadatas=document_metadata,
                embeddings=document_embeddings
            )
            
            logger.info(f"Successfully added {batch_size} documents to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            raise RuntimeError(f"Document addition failed: {e}") from e
    
    def search_similar_documents(self, 
                               query_embedding: List[float],
                               max_results: Optional[int] = None) -> Dict[str, List[Any]]:
        """
        Search for documents similar to the query embedding.
        
        Args:
            query_embedding: Vector embedding of the search query
            max_results: Maximum number of results to return (uses config default if None)
            
        Returns:
            Dictionary containing:
                - 'documents': List of matching document texts
                - 'metadatas': List of metadata for each match
                - 'distances': List of similarity distances (lower = more similar)
                
        Raises:
            RuntimeError: If ChromaDB search operation fails
        """
        if max_results is None:
            max_results = settings.top_k_results
        
        try:
            search_results = self.document_collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results,
                include=["documents", "metadatas", "distances"]
            )
            
            logger.debug(f"Retrieved {len(search_results.get('documents', [[]])[0])} similar documents")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search vector store: {e}")
            raise RuntimeError(f"Document search failed: {e}") from e
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the document collection.
        
        Returns:
            Dictionary with collection statistics including document count
        """
        try:
            document_count = self.document_collection.count()
            
            return {
                "collection_name": self.collection_name,
                "total_documents": document_count,
                "vector_store_path": str(settings.vector_store_directory)
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                "collection_name": self.collection_name,
                "total_documents": 0,
                "error": str(e)
            }
    
    def clear_collection(self) -> None:
        """
        Clear all documents from the collection.
        
        Warning: This operation is irreversible!
        """
        try:
            # Delete and recreate collection to clear all data
            self.chroma_client.delete_collection(name=self.collection_name)
            self.document_collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name
            )
            
            logger.warning(f"Cleared all documents from collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise RuntimeError(f"Collection clearing failed: {e}") from e
