from typing import List
import time
import logging
from ..config import (
    PROVIDER, GEMINI_API_KEY, OPENAI_API_KEY, 
    MAX_RETRIES, RETRY_BASE_DELAY, RETRY_MAX_DELAY, API_TIMEOUT
)

# Lazy imports to keep cold start slim
logger = logging.getLogger(__name__)

def retry_with_exponential_backoff(func, max_retries=MAX_RETRIES):
    """Decorator to retry function calls with exponential backoff"""
    def wrapper(*args, **kwargs):
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                
                # Check if it's a retryable error
                if any(keyword in error_msg for keyword in [
                    'timeout', 'deadline exceeded', '504', '503', '502', '500',
                    'rate limit', 'quota', 'temporarily unavailable'
                ]):
                    if attempt < max_retries:
                        delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                        logger.warning(f"API error on attempt {attempt + 1}/{max_retries + 1}: {e}")
                        logger.info(f"Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                        continue
                
                # Non-retryable error, raise immediately
                raise e
        
        # All retries exhausted
        logger.error(f"All {max_retries + 1} attempts failed. Last error: {last_exception}")
        raise last_exception
    
    return wrapper

def get_embedding_fn():
    if PROVIDER == "gemini":
        import google.generativeai as genai
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set")
        genai.configure(api_key=GEMINI_API_KEY)
        @retry_with_exponential_backoff
        def _embed_single(text: str) -> List[float]:
            """Embed a single text with retry logic"""
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        
        def _embed(texts: List[str]) -> List[List[float]]:
            # Google SDK supports batch embeddings
            embeddings = []
            for i, text in enumerate(texts):
                logger.debug(f"Embedding text {i+1}/{len(texts)}")
                embedding = _embed_single(text)
                embeddings.append(embedding)
                # Small delay between individual embeddings to avoid rate limits
                if i < len(texts) - 1:
                    time.sleep(0.1)
            return embeddings
        return _embed
    elif PROVIDER == "openai":
        from openai import OpenAI
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        @retry_with_exponential_backoff
        def _embed(texts: List[str]) -> List[List[float]]:
            logger.debug(f"Embedding batch of {len(texts)} texts with OpenAI")
            out = client.embeddings.create(
                model="text-embedding-3-small", 
                input=texts,
                timeout=API_TIMEOUT
            )
            return [d.embedding for d in out.data]
        return _embed
    else:
        raise RuntimeError(f"Unsupported PROVIDER: {PROVIDER}")
