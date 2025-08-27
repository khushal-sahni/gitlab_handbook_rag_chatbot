import os
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1200))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
TOP_K = int(os.getenv("TOP_K", 4))
MIN_SCORE = float(os.getenv("MIN_SCORE", 0.78))

# Ingestion performance settings
MAX_PAGES = int(os.getenv("MAX_PAGES", 50))  # Increased from 5
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 64))  # Increased from 32
CRAWL_DELAY = float(os.getenv("CRAWL_DELAY", 0.1))  # Reduced from 0.3
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 30))  # Increased from 20

# API retry settings
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 5))
RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", 1.0))
RETRY_MAX_DELAY = float(os.getenv("RETRY_MAX_DELAY", 60.0))
API_TIMEOUT = int(os.getenv("API_TIMEOUT", 120))

# Progress persistence
CHECKPOINT_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", 100))  # Save progress every N chunks
RESUME_FROM_CHECKPOINT = os.getenv("RESUME_FROM_CHECKPOINT", "true").lower() == "true"

SEED_URLS = [
    "https://about.gitlab.com/handbook/",
    "https://about.gitlab.com/direction/",
]

ALLOWED_NETLOC = {"about.gitlab.com"}
USER_AGENT = "gitlab-rag-bot/1.0 (+https://example)"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
FEEDBACK_CSV = os.path.join(LOG_DIR, "feedback.csv")
