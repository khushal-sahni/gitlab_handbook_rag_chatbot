"""
User feedback collection and management.

This module provides functionality for collecting, storing, and analyzing
user feedback on chatbot responses for continuous improvement.
"""

import csv
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..config.settings import settings

logger = logging.getLogger(__name__)


class UserFeedbackCollector:
    """
    Manages collection and storage of user feedback on chatbot responses.
    
    Provides functionality to collect user ratings, comments, and response
    metadata for analysis and system improvement.
    """
    
    FEEDBACK_RATINGS = {
        "helpful": 1,
        "not_helpful": 0,
        "very_helpful": 2
    }
    
    def __init__(self, feedback_file_path: Optional[Path] = None):
        """
        Initialize feedback collector with storage configuration.
        
        Args:
            feedback_file_path: Path to CSV file for storing feedback (uses config default if None)
        """
        self.feedback_file_path = feedback_file_path or settings.feedback_csv_path
        self._initialize_feedback_file()
        
        logger.debug(f"Initialized UserFeedbackCollector with file: {self.feedback_file_path}")
    
    def _initialize_feedback_file(self) -> None:
        """Initialize feedback CSV file with headers if it doesn't exist."""
        try:
            # Ensure feedback directory exists
            self.feedback_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create file with headers if it doesn't exist
            if not self.feedback_file_path.exists():
                with open(self.feedback_file_path, 'w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        "timestamp", "user_question", "chatbot_response", 
                        "feedback_rating", "source_urls", "response_time_seconds",
                        "similarity_scores", "user_comment"
                    ])
                logger.info(f"Created new feedback file: {self.feedback_file_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize feedback file: {e}")
            raise RuntimeError(f"Feedback file initialization failed: {e}") from e
    
    def record_feedback(self,
                       user_question: str,
                       chatbot_response: str,
                       feedback_rating: str,
                       source_urls: List[str],
                       response_time_seconds: float = 0.0,
                       similarity_scores: Optional[List[float]] = None,
                       user_comment: str = "") -> None:
        """
        Record user feedback for a chatbot interaction.
        
        Args:
            user_question: The user's original question
            chatbot_response: The chatbot's response
            feedback_rating: User rating ("helpful", "not_helpful", "very_helpful")
            source_urls: List of source URLs used in the response
            response_time_seconds: Time taken to generate response
            similarity_scores: Similarity scores for retrieved documents
            user_comment: Optional user comment
            
        Raises:
            ValueError: If feedback_rating is not valid
            RuntimeError: If feedback recording fails
        """
        # Validate feedback rating
        if feedback_rating not in self.FEEDBACK_RATINGS:
            raise ValueError(
                f"Invalid feedback rating: {feedback_rating}. "
                f"Valid options: {list(self.FEEDBACK_RATINGS.keys())}"
            )
        
        try:
            # Prepare feedback data
            timestamp = datetime.utcnow().isoformat()
            numeric_rating = self.FEEDBACK_RATINGS[feedback_rating]
            sources_string = ";".join(source_urls) if source_urls else ""
            scores_string = ";".join(map(str, similarity_scores)) if similarity_scores else ""
            
            # Write feedback to CSV
            with open(self.feedback_file_path, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([
                    timestamp, user_question, chatbot_response,
                    numeric_rating, sources_string, response_time_seconds,
                    scores_string, user_comment
                ])
            
            logger.info(f"Recorded {feedback_rating} feedback for question: {user_question[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
            raise RuntimeError(f"Feedback recording failed: {e}") from e
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about collected feedback.
        
        Returns:
            Dictionary containing feedback statistics and metrics
        """
        try:
            if not self.feedback_file_path.exists():
                return {"total_feedback": 0, "error": "No feedback file found"}
            
            feedback_data = []
            
            with open(self.feedback_file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                feedback_data = list(reader)
            
            if not feedback_data:
                return {"total_feedback": 0}
            
            # Calculate statistics
            total_feedback = len(feedback_data)
            ratings = [int(row.get('feedback_rating', 0)) for row in feedback_data]
            
            helpful_count = sum(1 for rating in ratings if rating >= 1)
            not_helpful_count = sum(1 for rating in ratings if rating == 0)
            very_helpful_count = sum(1 for rating in ratings if rating == 2)
            
            average_rating = sum(ratings) / len(ratings) if ratings else 0
            
            # Response time statistics
            response_times = []
            for row in feedback_data:
                try:
                    response_time = float(row.get('response_time_seconds', 0))
                    if response_time > 0:
                        response_times.append(response_time)
                except (ValueError, TypeError):
                    continue
            
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            return {
                "total_feedback": total_feedback,
                "helpful_count": helpful_count,
                "not_helpful_count": not_helpful_count,
                "very_helpful_count": very_helpful_count,
                "helpfulness_percentage": (helpful_count / total_feedback * 100) if total_feedback > 0 else 0,
                "average_rating": average_rating,
                "average_response_time_seconds": avg_response_time,
                "feedback_file_path": str(self.feedback_file_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to get feedback statistics: {e}")
            return {"total_feedback": 0, "error": str(e)}
    
    def export_feedback_data(self, export_file_path: Path) -> None:
        """
        Export feedback data to a new file for analysis.
        
        Args:
            export_file_path: Path where to export the feedback data
            
        Raises:
            RuntimeError: If export operation fails
        """
        try:
            if not self.feedback_file_path.exists():
                raise RuntimeError("No feedback data to export")
            
            # Copy feedback file to export location
            import shutil
            shutil.copy2(self.feedback_file_path, export_file_path)
            
            logger.info(f"Exported feedback data to: {export_file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export feedback data: {e}")
            raise RuntimeError(f"Feedback export failed: {e}") from e
    
    def get_recent_feedback(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent feedback entries.
        
        Args:
            limit: Maximum number of recent entries to return
            
        Returns:
            List of recent feedback entries
        """
        try:
            if not self.feedback_file_path.exists():
                return []
            
            with open(self.feedback_file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                all_feedback = list(reader)
            
            # Return most recent entries (CSV is chronologically ordered)
            recent_feedback = all_feedback[-limit:] if all_feedback else []
            recent_feedback.reverse()  # Most recent first
            
            return recent_feedback
            
        except Exception as e:
            logger.error(f"Failed to get recent feedback: {e}")
            return []


# Global feedback collector instance
feedback_collector = UserFeedbackCollector()
