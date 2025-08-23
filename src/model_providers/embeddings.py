from typing import List
from ..config import PROVIDER, GEMINI_API_KEY, OPENAI_API_KEY

# Lazy imports to keep cold start slim

def get_embedding_fn():
    if PROVIDER == "gemini":
        import google.generativeai as genai
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set")
        genai.configure(api_key=GEMINI_API_KEY)
        def _embed(texts: List[str]) -> List[List[float]]:
            # Google SDK supports batch embeddings
            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            return embeddings
        return _embed
    elif PROVIDER == "openai":
        from openai import OpenAI
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        client = OpenAI(api_key=OPENAI_API_KEY)
        def _embed(texts: List[str]) -> List[List[float]]:
            out = client.embeddings.create(model="text-embedding-3-small", input=texts)
            return [d.embedding for d in out.data]
        return _embed
    else:
        raise RuntimeError(f"Unsupported PROVIDER: {PROVIDER}")
