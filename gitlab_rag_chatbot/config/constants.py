"""
Application constants and default values.

This module contains all hardcoded constants used throughout the application.
These values should not be changed during runtime.
"""

from typing import Set, List

# Document processing defaults
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_MAX_PAGES_TO_CRAWL = 50
DEFAULT_EMBEDDING_BATCH_SIZE = 64

# Retrieval defaults  
DEFAULT_TOP_K_RESULTS = 4
DEFAULT_MIN_SIMILARITY_SCORE = 0.25

# Network and API defaults
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30
DEFAULT_CRAWL_DELAY_SECONDS = 0.1
DEFAULT_API_TIMEOUT_SECONDS = 120
DEFAULT_MAX_API_RETRIES = 5
DEFAULT_RETRY_BASE_DELAY_SECONDS = 1.0
DEFAULT_RETRY_MAX_DELAY_SECONDS = 60.0

# Checkpoint and persistence defaults
DEFAULT_CHECKPOINT_INTERVAL = 100
DEFAULT_RESUME_FROM_CHECKPOINT = True

# Supported embedding providers
SUPPORTED_EMBEDDING_PROVIDERS: Set[str] = {"gemini", "openai"}

# GitLab documentation domains
GITLAB_HANDBOOK_DOMAINS: Set[str] = {"about.gitlab.com"}

# GitLab documentation seed URLs
GITLAB_SEED_URLS: List[str] = [
    "https://about.gitlab.com/handbook/",
    "https://about.gitlab.com/direction/",
]

# File extensions to exclude from crawling
EXCLUDED_FILE_EXTENSIONS: tuple = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", 
    ".pdf", ".zip", ".tar", ".gz", ".bz2"
)

# User agent for web requests
DEFAULT_USER_AGENT = "gitlab-rag-chatbot/1.0 (+https://gitlab.com/rag-bot)"

# Directory names
DATA_DIRECTORY_NAME = "data"
CACHE_DIRECTORY_NAME = "cache" 
VECTOR_STORE_DIRECTORY_NAME = "chroma"
LOGS_DIRECTORY_NAME = "logs"

# File names
FEEDBACK_CSV_FILENAME = "user_feedback.csv"
INGESTION_LOG_FILENAME = "document_ingestion.log"
CHECKPOINT_FILENAME = "ingestion_checkpoint.json"

# Memory monitoring thresholds
HIGH_MEMORY_USAGE_THRESHOLD_PERCENT = 80.0
MEMORY_CLEANUP_INTERVAL_CHUNKS = 50
