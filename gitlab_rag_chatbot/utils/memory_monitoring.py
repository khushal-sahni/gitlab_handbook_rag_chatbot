"""
System memory monitoring and management utilities.

This module provides comprehensive memory monitoring capabilities for
tracking resource usage during document ingestion and processing.
"""

import gc
import logging
from typing import Dict, Any, Optional
import psutil

from ..config.constants import HIGH_MEMORY_USAGE_THRESHOLD_PERCENT

logger = logging.getLogger(__name__)


class SystemMemoryMonitor:
    """
    Comprehensive system memory monitoring and management.
    
    Provides utilities for tracking memory usage, detecting high usage conditions,
    and performing memory cleanup operations during resource-intensive operations.
    """
    
    def __init__(self, high_usage_threshold: float = HIGH_MEMORY_USAGE_THRESHOLD_PERCENT):
        """
        Initialize memory monitor with configurable threshold.
        
        Args:
            high_usage_threshold: Memory usage percentage that triggers warnings (0-100)
        """
        self.high_usage_threshold = high_usage_threshold
        self.process = psutil.Process()
        logger.debug(f"Initialized memory monitor with {high_usage_threshold}% threshold")
    
    def get_current_memory_usage(self) -> Dict[str, float]:
        """
        Get comprehensive current memory usage statistics.
        
        Returns:
            Dictionary containing memory usage metrics:
                - resident_set_size_mb: Physical memory currently used
                - virtual_memory_size_mb: Virtual memory used
                - memory_percentage: Percentage of system memory used
                - available_system_memory_mb: Available system memory
                - total_system_memory_mb: Total system memory
        """
        try:
            # Process-specific memory info
            process_memory_info = self.process.memory_info()
            process_memory_percent = self.process.memory_percent()
            
            # System-wide memory info
            system_memory_info = psutil.virtual_memory()
            
            return {
                "resident_set_size_mb": process_memory_info.rss / 1024 / 1024,
                "virtual_memory_size_mb": process_memory_info.vms / 1024 / 1024,
                "memory_percentage": process_memory_percent,
                "available_system_memory_mb": system_memory_info.available / 1024 / 1024,
                "total_system_memory_mb": system_memory_info.total / 1024 / 1024,
                "system_memory_percentage": system_memory_info.percent
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return {
                "resident_set_size_mb": 0.0,
                "virtual_memory_size_mb": 0.0,
                "memory_percentage": 0.0,
                "available_system_memory_mb": 0.0,
                "total_system_memory_mb": 0.0,
                "system_memory_percentage": 0.0
            }
    
    def log_memory_usage(self, operation_context: str) -> Dict[str, float]:
        """
        Log current memory usage with contextual information.
        
        Args:
            operation_context: Description of current operation for context
            
        Returns:
            Current memory usage statistics
        """
        memory_stats = self.get_current_memory_usage()
        
        logger.info(
            f"MEMORY [{operation_context}] - "
            f"RSS: {memory_stats['resident_set_size_mb']:.1f}MB, "
            f"VMS: {memory_stats['virtual_memory_size_mb']:.1f}MB, "
            f"Process: {memory_stats['memory_percentage']:.1f}%, "
            f"System: {memory_stats['system_memory_percentage']:.1f}%, "
            f"Available: {memory_stats['available_system_memory_mb']:.1f}MB"
        )
        
        # Check for high memory usage
        if memory_stats['memory_percentage'] > self.high_usage_threshold:
            logger.warning(
                f"HIGH MEMORY USAGE DETECTED: {memory_stats['memory_percentage']:.1f}% "
                f"(threshold: {self.high_usage_threshold}%) - "
                f"Available: {memory_stats['available_system_memory_mb']:.1f}MB"
            )
        
        return memory_stats
    
    def perform_memory_cleanup(self, operation_context: str = "cleanup") -> Dict[str, float]:
        """
        Perform garbage collection and log memory cleanup results.
        
        Args:
            operation_context: Description of cleanup context
            
        Returns:
            Dictionary with before/after memory stats and cleanup results
        """
        # Get memory usage before cleanup
        memory_before_cleanup = self.get_current_memory_usage()
        
        # Force garbage collection
        objects_collected = gc.collect()
        
        # Get memory usage after cleanup
        memory_after_cleanup = self.get_current_memory_usage()
        
        # Calculate memory freed
        memory_freed_mb = (
            memory_before_cleanup['resident_set_size_mb'] - 
            memory_after_cleanup['resident_set_size_mb']
        )
        
        logger.info(
            f"MEMORY CLEANUP [{operation_context}] - "
            f"Objects collected: {objects_collected}, "
            f"Memory freed: {memory_freed_mb:.1f}MB, "
            f"Before: {memory_before_cleanup['resident_set_size_mb']:.1f}MB, "
            f"After: {memory_after_cleanup['resident_set_size_mb']:.1f}MB"
        )
        
        return {
            "objects_collected": objects_collected,
            "memory_freed_mb": memory_freed_mb,
            "memory_before_mb": memory_before_cleanup['resident_set_size_mb'],
            "memory_after_mb": memory_after_cleanup['resident_set_size_mb']
        }
    
    def is_memory_usage_high(self) -> bool:
        """
        Check if current memory usage exceeds the high usage threshold.
        
        Returns:
            True if memory usage is above threshold, False otherwise
        """
        current_usage = self.get_current_memory_usage()
        return current_usage['memory_percentage'] > self.high_usage_threshold
    
    def get_memory_usage_summary(self) -> str:
        """
        Get a human-readable summary of current memory usage.
        
        Returns:
            Formatted string with memory usage summary
        """
        memory_stats = self.get_current_memory_usage()
        
        return (
            f"Memory Usage Summary: "
            f"Process RSS: {memory_stats['resident_set_size_mb']:.1f}MB "
            f"({memory_stats['memory_percentage']:.1f}%), "
            f"System: {memory_stats['system_memory_percentage']:.1f}% used, "
            f"{memory_stats['available_system_memory_mb']:.1f}MB available"
        )
    
    def monitor_operation(self, operation_name: str):
        """
        Context manager for monitoring memory usage during operations.
        
        Args:
            operation_name: Name of the operation being monitored
            
        Usage:
            with memory_monitor.monitor_operation("document_processing"):
                # Your operation here
                pass
        """
        return MemoryOperationMonitor(self, operation_name)


class MemoryOperationMonitor:
    """Context manager for monitoring memory usage during specific operations."""
    
    def __init__(self, memory_monitor: SystemMemoryMonitor, operation_name: str):
        """
        Initialize operation monitor.
        
        Args:
            memory_monitor: SystemMemoryMonitor instance
            operation_name: Name of the operation being monitored
        """
        self.memory_monitor = memory_monitor
        self.operation_name = operation_name
        self.start_memory_stats = None
    
    def __enter__(self):
        """Start monitoring the operation."""
        self.start_memory_stats = self.memory_monitor.log_memory_usage(f"START_{self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End monitoring and log final statistics."""
        end_memory_stats = self.memory_monitor.log_memory_usage(f"END_{self.operation_name}")
        
        if self.start_memory_stats:
            memory_delta = (
                end_memory_stats['resident_set_size_mb'] - 
                self.start_memory_stats['resident_set_size_mb']
            )
            
            logger.info(
                f"MEMORY OPERATION [{self.operation_name}] COMPLETED - "
                f"Memory delta: {memory_delta:+.1f}MB"
            )
        
        # Perform cleanup if memory usage is high
        if self.memory_monitor.is_memory_usage_high():
            self.memory_monitor.perform_memory_cleanup(f"POST_{self.operation_name}")


# Global memory monitor instance
system_memory_monitor = SystemMemoryMonitor()
