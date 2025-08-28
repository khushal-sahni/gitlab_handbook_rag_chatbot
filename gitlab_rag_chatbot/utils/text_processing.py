"""
Text processing utilities for document chunking and content preparation.

This module provides robust text processing capabilities including intelligent
document chunking with fallback mechanisms and content normalization.
"""

import logging
from typing import List, Optional

from ..config.settings import settings

logger = logging.getLogger(__name__)


class DocumentTextSplitter:
    """
    Robust document text splitting with multiple chunking strategies.
    
    Provides intelligent text chunking using LangChain's RecursiveCharacterTextSplitter
    with a fallback to a simple chunking method for reliability.
    """
    
    def __init__(self, 
                 chunk_size: Optional[int] = None,
                 chunk_overlap: Optional[int] = None):
        """
        Initialize text splitter with configuration.
        
        Args:
            chunk_size: Maximum size of each text chunk (uses config default if None)
            chunk_overlap: Number of characters to overlap between chunks (uses config default if None)
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # Validate configuration
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        
        logger.debug(f"Initialized text splitter: chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks using the most appropriate method available.
        
        Attempts to use LangChain's RecursiveCharacterTextSplitter for optimal
        chunking, with fallback to a simple method if LangChain is unavailable.
        
        Args:
            text: Text content to split into chunks
            
        Returns:
            List of text chunks, with empty chunks filtered out
        """
        if not text or not text.strip():
            logger.debug("Empty or whitespace-only text provided")
            return []
        
        try:
            return self._split_with_langchain(text)
        except ImportError:
            logger.warning("LangChain not available, using simple chunking method")
            return self._split_with_simple_method(text)
        except Exception as e:
            logger.warning(f"LangChain chunking failed ({e}), using simple chunking method")
            return self._split_with_simple_method(text)
    
    def _split_with_langchain(self, text: str) -> List[str]:
        """
        Split text using LangChain's RecursiveCharacterTextSplitter.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
            
        Raises:
            ImportError: If LangChain is not available
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ". ", " ", ""]  # Hierarchical separators
        )
        
        chunks = text_splitter.split_text(text)
        filtered_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        logger.debug(f"LangChain chunking produced {len(filtered_chunks)} chunks")
        return filtered_chunks
    
    def _split_with_simple_method(self, text: str) -> List[str]:
        """
        Split text using a simple sliding window approach.
        
        This is a fallback method that ensures reliable chunking even when
        LangChain is not available.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []
        
        chunks = []
        current_position = 0
        text_length = len(text)
        
        while current_position < text_length:
            # Calculate chunk end position
            chunk_end_position = min(current_position + self.chunk_size, text_length)
            
            # Extract chunk
            chunk_text = text[current_position:chunk_end_position].strip()
            
            # Add non-empty chunks
            if chunk_text:
                chunks.append(chunk_text)
            
            # Calculate next starting position with overlap
            next_position = chunk_end_position - self.chunk_overlap
            
            # Ensure forward progress to prevent infinite loops
            if next_position <= current_position:
                next_position = current_position + 1
            
            current_position = next_position
        
        logger.debug(f"Simple chunking produced {len(chunks)} chunks")
        return chunks
    
    def get_chunk_statistics(self, chunks: List[str]) -> dict:
        """
        Calculate statistics about the generated chunks.
        
        Args:
            chunks: List of text chunks to analyze
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "average_chunk_length": 0.0,
                "min_chunk_length": 0,
                "max_chunk_length": 0,
                "total_characters": 0
            }
        
        chunk_lengths = [len(chunk) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "average_chunk_length": sum(chunk_lengths) / len(chunk_lengths),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "total_characters": sum(chunk_lengths)
        }
    
    def validate_chunks(self, chunks: List[str]) -> List[str]:
        """
        Validate and clean chunks, removing any that are too small or empty.
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            List of validated chunks
        """
        min_chunk_length = max(10, self.chunk_overlap // 2)  # Minimum meaningful chunk size
        
        validated_chunks = []
        for i, chunk in enumerate(chunks):
            cleaned_chunk = chunk.strip()
            
            if len(cleaned_chunk) >= min_chunk_length:
                validated_chunks.append(cleaned_chunk)
            else:
                logger.debug(f"Skipping chunk {i}: too short ({len(cleaned_chunk)} chars)")
        
        logger.debug(f"Validated {len(validated_chunks)} out of {len(chunks)} chunks")
        return validated_chunks


# Convenience functions for backward compatibility
def chunk_text(text: str, 
               size: Optional[int] = None, 
               overlap: Optional[int] = None) -> List[str]:
    """
    Convenience function for splitting text into chunks.
    
    This function provides backward compatibility with existing code
    while using the improved DocumentTextSplitter class.
    
    Args:
        text: Text to split into chunks
        size: Chunk size (uses config default if None)
        overlap: Chunk overlap (uses config default if None)
        
    Returns:
        List of text chunks
    """
    splitter = DocumentTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_text_into_chunks(text)
