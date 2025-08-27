"""
Checkpoint system for ingestion progress persistence.
Allows resuming from where the process left off in case of failures.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Set, Any, Optional
from .config import DATA_DIR

logger = logging.getLogger(__name__)

CHECKPOINT_FILE = os.path.join(DATA_DIR, "ingestion_checkpoint.json")

class IngestionCheckpoint:
    """Manages checkpoint state for ingestion process"""
    
    def __init__(self):
        self.state = {
            "version": "1.0",
            "created_at": None,
            "updated_at": None,
            "processed_urls": [],
            "seen_urls": [],
            "current_page": 0,
            "total_chunks_processed": 0,
            "last_batch_id": 0,
            "queue_urls": [],
            "config_snapshot": {}
        }
        self.load()
    
    def load(self) -> bool:
        """Load checkpoint from disk if it exists"""
        if not os.path.exists(CHECKPOINT_FILE):
            logger.info("No checkpoint file found, starting fresh")
            return False
        
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                self.state = json.load(f)
            logger.info(f"Loaded checkpoint from {self.state.get('updated_at')}")
            logger.info(f"Resume state: {self.state['total_chunks_processed']} chunks, "
                       f"{len(self.state['processed_urls'])} pages processed")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def save(self):
        """Save current state to checkpoint file"""
        try:
            self.state["updated_at"] = datetime.now().isoformat()
            if not self.state["created_at"]:
                self.state["created_at"] = self.state["updated_at"]
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
            
            # Write to temporary file first, then rename (atomic operation)
            temp_file = CHECKPOINT_FILE + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(self.state, f, indent=2)
            os.rename(temp_file, CHECKPOINT_FILE)
            
            logger.debug(f"Checkpoint saved: {self.state['total_chunks_processed']} chunks processed")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def clear(self):
        """Remove checkpoint file and reset state"""
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            logger.info("Checkpoint cleared")
        self.__init__()
    
    def update_progress(self, 
                       processed_url: Optional[str] = None,
                       chunks_count: int = 0,
                       seen_urls: Optional[Set[str]] = None,
                       queue_urls: Optional[List[str]] = None):
        """Update progress information"""
        
        if processed_url and processed_url not in self.state["processed_urls"]:
            self.state["processed_urls"].append(processed_url)
            self.state["current_page"] = len(self.state["processed_urls"])
        
        if chunks_count > 0:
            self.state["total_chunks_processed"] += chunks_count
        
        if seen_urls is not None:
            self.state["seen_urls"] = list(seen_urls)
        
        if queue_urls is not None:
            self.state["queue_urls"] = queue_urls
    
    def should_skip_url(self, url: str) -> bool:
        """Check if URL was already processed"""
        return url in self.state["processed_urls"]
    
    def get_seen_urls(self) -> Set[str]:
        """Get set of URLs that have been seen before"""
        return set(self.state["seen_urls"])
    
    def get_queue_urls(self) -> List[str]:
        """Get list of URLs still in queue"""
        return self.state["queue_urls"].copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current progress statistics"""
        return {
            "pages_processed": len(self.state["processed_urls"]),
            "chunks_processed": self.state["total_chunks_processed"],
            "urls_seen": len(self.state["seen_urls"]),
            "queue_size": len(self.state["queue_urls"]),
            "created_at": self.state["created_at"],
            "updated_at": self.state["updated_at"]
        }
    
    def set_config_snapshot(self, config: Dict[str, Any]):
        """Save configuration snapshot for consistency checking"""
        self.state["config_snapshot"] = config
    
    def validate_config(self, current_config: Dict[str, Any]) -> bool:
        """Validate that current config matches checkpoint config"""
        if not self.state["config_snapshot"]:
            return True
        
        # Check critical config values
        critical_keys = ["CHUNK_SIZE", "CHUNK_OVERLAP", "EMBEDDING_BATCH_SIZE"]
        for key in critical_keys:
            if (key in self.state["config_snapshot"] and 
                key in current_config and
                self.state["config_snapshot"][key] != current_config[key]):
                logger.warning(f"Config mismatch for {key}: "
                             f"checkpoint={self.state['config_snapshot'][key]}, "
                             f"current={current_config[key]}")
                return False
        return True
