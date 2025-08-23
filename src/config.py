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
