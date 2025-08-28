"""
Progress tracking and checkpointing for document ingestion.

This module provides robust progress tracking capabilities that allow
ingestion processes to resume from where they left off in case of failures.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Set, Any, Optional
from pathlib import Path

from ...config.settings import settings

logger = logging.getLogger(__name__)


class IngestionProgressTracker:
    """
    Manages checkpoint state and progress tracking for document ingestion.
    
    This class provides persistent storage of ingestion progress, allowing
    processes to resume from the last successful checkpoint in case of failures.
    """
    
    CHECKPOINT_VERSION = "2.0"
    
    def __init__(self, checkpoint_file_path: Optional[Path] = None):
        """
        Initialize progress tracker with checkpoint file.
        
        Args:
            checkpoint_file_path: Path to checkpoint file (uses config default if None)
        """
        self.checkpoint_file_path = checkpoint_file_path or settings.checkpoint_file_path
        self._initialize_state()
        self._load_checkpoint()
    
    def _initialize_state(self) -> None:
        """Initialize default checkpoint state."""
        self.checkpoint_state = {
            "version": self.CHECKPOINT_VERSION,
            "created_at": None,
            "updated_at": None,
            "processed_urls": [],
            "discovered_urls": [],
            "current_page_index": 0,
            "total_chunks_processed": 0,
            "last_batch_identifier": 0,
            "pending_urls_queue": [],
            "configuration_snapshot": {},
            "ingestion_statistics": {
                "total_pages_crawled": 0,
                "total_documents_processed": 0,
                "total_embedding_batches": 0,
                "average_chunks_per_page": 0.0,
                "last_processing_duration_seconds": 0.0
            }
        }
    
    def _load_checkpoint(self) -> bool:
        """
        Load checkpoint from disk if it exists.
        
        Returns:
            True if checkpoint was loaded successfully, False otherwise
        """
        if not self.checkpoint_file_path.exists():
            logger.info("No checkpoint file found - starting fresh ingestion")
            return False
        
        try:
            with open(self.checkpoint_file_path, 'r', encoding='utf-8') as file:
                loaded_state = json.load(file)
            
            # Validate checkpoint version compatibility
            if loaded_state.get("version") != self.CHECKPOINT_VERSION:
                logger.warning(
                    f"Checkpoint version mismatch. Expected {self.CHECKPOINT_VERSION}, "
                    f"found {loaded_state.get('version')}. Starting fresh."
                )
                return False
            
            self.checkpoint_state = loaded_state
            
            logger.info(f"Loaded checkpoint from {self.checkpoint_state.get('updated_at')}")
            logger.info(
                f"Resume state: {self.checkpoint_state['total_chunks_processed']} chunks, "
                f"{len(self.checkpoint_state['processed_urls'])} pages processed"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Starting fresh ingestion due to checkpoint loading failure")
            return False
    
    def save_checkpoint(self) -> None:
        """
        Save current progress state to checkpoint file.
        
        Uses atomic write operation to prevent corruption.
        """
        try:
            # Update timestamps
            current_time = datetime.now().isoformat()
            self.checkpoint_state["updated_at"] = current_time
            
            if not self.checkpoint_state["created_at"]:
                self.checkpoint_state["created_at"] = current_time
            
            # Ensure checkpoint directory exists
            self.checkpoint_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Atomic write: write to temporary file, then rename
            temporary_file_path = self.checkpoint_file_path.with_suffix('.tmp')
            
            with open(temporary_file_path, 'w', encoding='utf-8') as file:
                json.dump(self.checkpoint_state, file, indent=2, ensure_ascii=False)
            
            # Atomic rename operation
            temporary_file_path.rename(self.checkpoint_file_path)
            
            logger.debug(
                f"Checkpoint saved: {self.checkpoint_state['total_chunks_processed']} "
                f"chunks processed"
            )
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise RuntimeError(f"Checkpoint save operation failed: {e}") from e
    
    def clear_checkpoint(self) -> None:
        """
        Remove checkpoint file and reset state.
        
        This should be called when ingestion completes successfully.
        """
        try:
            if self.checkpoint_file_path.exists():
                self.checkpoint_file_path.unlink()
                logger.info("Checkpoint file removed after successful completion")
            
            self._initialize_state()
            
        except Exception as e:
            logger.error(f"Failed to clear checkpoint: {e}")
    
    def update_progress(self, 
                       processed_url: Optional[str] = None,
                       chunks_processed_count: int = 0,
                       discovered_urls: Optional[Set[str]] = None,
                       pending_urls_queue: Optional[List[str]] = None,
                       processing_duration_seconds: float = 0.0) -> None:
        """
        Update progress information in checkpoint state.
        
        Args:
            processed_url: URL that was just processed (if any)
            chunks_processed_count: Number of chunks processed in this update
            discovered_urls: Set of all discovered URLs
            pending_urls_queue: List of URLs still pending processing
            processing_duration_seconds: Time taken for last processing operation
        """
        # Update processed URL
        if processed_url and processed_url not in self.checkpoint_state["processed_urls"]:
            self.checkpoint_state["processed_urls"].append(processed_url)
            self.checkpoint_state["current_page_index"] = len(self.checkpoint_state["processed_urls"])
            self.checkpoint_state["ingestion_statistics"]["total_pages_crawled"] += 1
        
        # Update chunk count
        if chunks_processed_count > 0:
            self.checkpoint_state["total_chunks_processed"] += chunks_processed_count
            self.checkpoint_state["ingestion_statistics"]["total_documents_processed"] += chunks_processed_count
        
        # Update discovered URLs
        if discovered_urls is not None:
            self.checkpoint_state["discovered_urls"] = list(discovered_urls)
        
        # Update pending queue
        if pending_urls_queue is not None:
            self.checkpoint_state["pending_urls_queue"] = pending_urls_queue
        
        # Update processing duration
        if processing_duration_seconds > 0:
            self.checkpoint_state["ingestion_statistics"]["last_processing_duration_seconds"] = processing_duration_seconds
        
        # Calculate average chunks per page
        total_pages = self.checkpoint_state["ingestion_statistics"]["total_pages_crawled"]
        total_chunks = self.checkpoint_state["total_chunks_processed"]
        
        if total_pages > 0:
            self.checkpoint_state["ingestion_statistics"]["average_chunks_per_page"] = total_chunks / total_pages
    
    def should_skip_url(self, url: str) -> bool:
        """
        Check if URL was already processed and should be skipped.
        
        Args:
            url: URL to check
            
        Returns:
            True if URL should be skipped, False if it should be processed
        """
        return url in self.checkpoint_state["processed_urls"]
    
    def get_discovered_urls(self) -> Set[str]:
        """
        Get set of URLs that have been discovered during crawling.
        
        Returns:
            Set of discovered URLs
        """
        return set(self.checkpoint_state["discovered_urls"])
    
    def get_pending_urls_queue(self) -> List[str]:
        """
        Get list of URLs still pending processing.
        
        Returns:
            Copy of pending URLs queue
        """
        return self.checkpoint_state["pending_urls_queue"].copy()
    
    def get_progress_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive progress statistics.
        
        Returns:
            Dictionary containing detailed progress information
        """
        stats = self.checkpoint_state["ingestion_statistics"].copy()
        stats.update({
            "pages_processed": len(self.checkpoint_state["processed_urls"]),
            "total_chunks_processed": self.checkpoint_state["total_chunks_processed"],
            "urls_discovered": len(self.checkpoint_state["discovered_urls"]),
            "urls_pending": len(self.checkpoint_state["pending_urls_queue"]),
            "checkpoint_created_at": self.checkpoint_state["created_at"],
            "checkpoint_updated_at": self.checkpoint_state["updated_at"],
            "completion_percentage": self._calculate_completion_percentage()
        })
        return stats
    
    def _calculate_completion_percentage(self) -> float:
        """
        Calculate estimated completion percentage.
        
        Returns:
            Completion percentage (0.0 to 100.0)
        """
        total_discovered = len(self.checkpoint_state["discovered_urls"])
        total_processed = len(self.checkpoint_state["processed_urls"])
        
        if total_discovered == 0:
            return 0.0
        
        return min(100.0, (total_processed / total_discovered) * 100.0)
    
    def set_configuration_snapshot(self, configuration: Dict[str, Any]) -> None:
        """
        Save configuration snapshot for consistency validation.
        
        Args:
            configuration: Dictionary of configuration values to snapshot
        """
        self.checkpoint_state["configuration_snapshot"] = configuration.copy()
    
    def validate_configuration_consistency(self, current_configuration: Dict[str, Any]) -> bool:
        """
        Validate that current configuration matches checkpoint configuration.
        
        Args:
            current_configuration: Current configuration to validate
            
        Returns:
            True if configurations are consistent, False otherwise
        """
        if not self.checkpoint_state["configuration_snapshot"]:
            return True
        
        # Critical configuration keys that must match
        critical_configuration_keys = [
            "chunk_size", "chunk_overlap", "embedding_batch_size", "embedding_provider"
        ]
        
        saved_config = self.checkpoint_state["configuration_snapshot"]
        
        for key in critical_configuration_keys:
            if (key in saved_config and 
                key in current_configuration and
                saved_config[key] != current_configuration[key]):
                
                logger.warning(
                    f"Configuration mismatch for {key}: "
                    f"checkpoint={saved_config[key]}, current={current_configuration[key]}"
                )
                return False
        
        return True
