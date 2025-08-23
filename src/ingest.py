import os, sys, queue
from urllib.parse import urlparse
from .config import (
    SEED_URLS, CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_DIR, MIN_SCORE, DATA_DIR
)
from .utils import fetch, extract_links, html_to_text, chunk_text
from .model_providers.embeddings import get_embedding_fn
from .retriever import Retriever

os.makedirs(CHROMA_DIR, exist_ok=True)

MAX_PAGES = 250  # keep it bounded for weekend run


def crawl(seed_urls):
    seen = set()
    q = queue.Queue()
    for u in seed_urls:
        q.put(u)
        seen.add(u)

    pages = []
    while not q.empty() and len(pages) < MAX_PAGES:
        url = q.get()
        try:
            html = fetch(url)
        except Exception as e:
            print("fetch failed:", url, e)
            continue
        links = extract_links(url, html)
        for l in links:
            if l not in seen:
                seen.add(l)
                q.put(l)
        pages.append((url, html))
    return pages


def build_index():
    embed = get_embedding_fn()
    r = Retriever()

    pages = crawl(SEED_URLS)
    print(f"Crawled {len(pages)} pages")

    ids, docs, metas = [], [], []
    for idx, (url, html) in enumerate(pages):
        text = html_to_text(html)
        chunks = chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        for j, ch in enumerate(chunks):
            ids.append(f"{idx}-{j}")
            docs.append(ch)
            metas.append({"url": url})

    print(f"Prepared {len(docs)} chunks; embedding …")
    # Batch embed to avoid rate limits; tune batch size if needed
    B = 64
    embs = []
    for i in range(0, len(docs), B):
        batch = docs[i:i+B]
        embs.extend(embed(batch))
        print(f"embedded {i+len(batch)}/{len(docs)}")

    r.add(ids=ids, docs=docs, metas=metas, embs=embs)
    print("Index built and persisted to:", CHROMA_DIR)

if __name__ == "__main__":
    build_index()
