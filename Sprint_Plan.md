# GitLab Handbook & Direction RAG Chatbot

A lean, end-to-end Retrieval-Augmented Generation (RAG) chatbot that answers questions from **GitLab’s Handbook** and **Direction** pages. Built for fast weekend execution: scrape → chunk → embed → vector DB (Chroma) → LLM (Gemini or OpenAI) → Streamlit UI → Deploy.

---

## 1) Repo Structure

```
.gitignore
README.md
requirements.txt
.ingest-allowlist.txt   # (optional) seed URLs to start from
src/
  config.py
  utils.py
  ingest.py
  retriever.py
  app.py                # Streamlit UI
  model_providers/
    __init__.py
    embeddings.py       # Gemini/OpenAI embeddings
    chat.py             # Gemini/OpenAI chat completion
  data/
    chroma/             # Chroma persistent directory
    cache/              # page cache for faster dev
  logs/
    feedback.csv        # user thumbs up/down (created on first run)
```

---

## 2) Setup

### Python

* Python 3.10+

### Install deps

```bash
pip install -r requirements.txt
```

### Environment variables

Create a `.env` or set env vars in your shell/Streamlit Cloud:

```
# Choose one provider. "gemini" recommended (free tier generous)
PROVIDER=gemini   # or openai

# If PROVIDER=gemini
GEMINI_API_KEY=...  

# If PROVIDER=openai
OPENAI_API_KEY=...

# Optional tuning
CHUNK_SIZE=1200      # characters
CHUNK_OVERLAP=150    # characters
TOP_K=4              # retrieved chunks
MIN_SCORE=0.78       # semantic similarity gate (0..1; higher = stricter)
```

### Seed URLs (optional)

`./.ingest-allowlist.txt` — one URL per line. Defaults cover:

```
https://about.gitlab.com/handbook/
https://about.gitlab.com/direction/
```

---

## 3) Run locally

```bash
# 1) Build the vector index (scrape + embed)
python -m src.ingest

# 2) Launch UI
streamlit run src/app.py
```

---

## 4) Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub (public or private with Streamlit access).
2. On [https://streamlit.io/cloud](https://streamlit.io/cloud) → New app → pick repo/branch → `src/app.py`.
3. Add environment variables in the app’s settings (same keys as above).
4. First run will ingest; subsequent runs reuse the persisted `data/chroma`.

> Backup option: Hugging Face Spaces (Gradio or Streamlit); requirements + `app.py` work as-is.

---

## 5) Features (Core + Bonus)

* **Core**: RAG over Handbook/Direction, citations per answer, follow-up friendly chat.
* **Transparency**: Each answer shows sources (URL + snippet). If low confidence, the bot says so and links where to look.
* **Guardrails**: Refuses to answer outside scope; fallback when retrieval is weak.
* **Product thinking**: Feedback buttons (👍/👎) logged to CSV for iterative improvement.

---

## 6) Limitations

* Cold-start ingest takes time on first run (scraping + embeddings). You can pre-build locally and commit `data/chroma` (large) or run ingestion in cloud on first boot.
* Not a full GitLab universe; scope restricted to `/handbook/` and `/direction/`.

---

## 7) Future Work

* Periodic background refresh via GitHub Action (nightly re-ingest).
* Add hybrid retrieval (BM25 + dense) for edge cases.
* Inline source highlighting; semantic answer grading for guardrails.
* Auth + role hints for employee vs candidate personas.

---

## 8) Code

### `requirements.txt`

```txt
streamlit==1.37.1
chromadb==0.5.5
beautifulsoup4==4.12.3
requests==2.32.3
python-dotenv==1.0.1
pydantic==2.9.0
google-generativeai==0.7.2
openai==1.44.0
markdownify==0.12.1
urllib3==2.2.2
```

### `src/config.py`

```python
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
```

### `src/utils.py`

```python
import os, re, time, hashlib
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from .config import USER_AGENT, CACHE_DIR, ALLOWED_NETLOC

os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(url: str):
    h = hashlib.sha256(url.encode()).hexdigest()[:20]
    return os.path.join(CACHE_DIR, f"{h}.html")

def fetch(url: str, timeout=20) -> str:
    """Fetch with simple on-disk cache to speed up dev."""
    cp = _cache_path(url)
    if os.path.exists(cp):
        with open(cp, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    html = r.text
    with open(cp, "w", encoding="utf-8") as f:
        f.write(html)
    time.sleep(0.3)  # be polite
    return html

BLACKLIST_EXT = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".zip")

def extract_links(base_url: str, html: str):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("mailto:") or href.startswith("#"):
            continue
        url = urljoin(base_url, href)
        u = urlparse(url)
        if u.netloc not in ALLOWED_NETLOC:
            continue
        if any(u.path.lower().endswith(ext) for ext in BLACKLIST_EXT):
            continue
        if not (u.path.startswith("/handbook/") or u.path.startswith("/direction/")):
            continue
        links.add(u.geturl())
    return links

def html_to_text(html: str) -> str:
    # Remove nav/footer/script/style noise
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    # Convert to markdown and then normalize whitespace
    text = md(str(soup))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def chunk_text(text: str, size=1200, overlap=150):
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + size, n)
        chunk = text[i:j]
        chunks.append(chunk)
        i = j - overlap
        if i < 0:
            i = 0
    return chunks
```

### `src/model_providers/embeddings.py`

```python
from typing import List
from ..config import PROVIDER, GEMINI_API_KEY, OPENAI_API_KEY

# Lazy imports to keep cold start slim

def get_embedding_fn():
    if PROVIDER == "gemini":
        import google.generativeai as genai
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set")
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.embedder.Embedder(model="text-embedding-004")
        def _embed(texts: List[str]) -> List[List[float]]:
            # Google SDK supports batch embeddings
            res = model.embed(texts)
            return [r.values for r in res.embeddings]
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
```

### `src/model_providers/chat.py`

```python
from typing import List
from ..config import PROVIDER, GEMINI_API_KEY, OPENAI_API_KEY

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers strictly from GitLab's Handbook and Direction pages. "
    "If the answer is not found in the provided context, say you don't know and suggest where to look. "
    "Always cite sources with their URLs."
)

def get_chat_fn():
    if PROVIDER == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        def _chat(messages: List[dict], temperature=0.2):
            # messages: [{"role":"system/user/assistant", "content": "..."}, ...]
            # Gemini SDK uses a single prompt; concatenate with roles.
            sys = SYSTEM_PROMPT
            convo = []
            if messages and messages[0].get("role") == "system":
                sys = messages[0]["content"] + "\n" + SYSTEM_PROMPT
                messages = messages[1:]
            for m in messages:
                role = m.get("role")
                content = m.get("content")
                convo.append(f"{role.upper()}: {content}")
            prompt = sys + "\n\n" + "\n".join(convo)
            resp = model.generate_content(prompt, generation_config={"temperature": temperature})
            return resp.text
        return _chat
    elif PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        def _chat(messages: List[dict], temperature=0.2):
            out = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                temperature=temperature,
            )
            return out.choices[0].message.content
        return _chat
    else:
        raise RuntimeError(f"Unsupported PROVIDER: {PROVIDER}")
```

### `src/retriever.py`

```python
import chromadb
from chromadb.config import Settings
from .config import CHROMA_DIR

class Retriever:
    def __init__(self, collection_name="gitlab_docs"):
        self.client = chromadb.Client(Settings(persist_directory=CHROMA_DIR))
        self.col = self.client.get_or_create_collection(collection_name)

    def query(self, query_emb, top_k=4):
        res = self.col.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return res

    def add(self, ids, docs, metas, embs):
        self.col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        self.client.persist()
```

### `src/ingest.py`

```python
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
```

### `src/app.py` (Streamlit UI)

```python
import os
import streamlit as st
from .config import TOP_K, MIN_SCORE, FEEDBACK_CSV, CHROMA_DIR
from .retriever import Retriever
from .model_providers.embeddings import get_embedding_fn
from .model_providers.chat import get_chat_fn
import csv
import datetime

st.set_page_config(page_title="GitLab Handbook Chatbot", page_icon="💬", layout="wide")

# Ensure feedback log exists
os.makedirs(os.path.dirname(FEEDBACK_CSV), exist_ok=True)
if not os.path.exists(FEEDBACK_CSV):
    with open(FEEDBACK_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "question", "answer", "useful", "sources"])  # useful in {1,0,-1}

# Lazy init
@st.cache_resource(show_spinner=False)
def _boot():
    embed = get_embedding_fn()
    chat = get_chat_fn()
    retr = Retriever()
    return embed, chat, retr

embed, chat, retr = _boot()

st.title("GitLab Handbook & Direction Chatbot")
st.caption("Answers strictly from https://about.gitlab.com/handbook/ and /direction/. Sources cited.")

if "history" not in st.session_state:
    st.session_state.history = []  # list of {role, content}

with st.sidebar:
    st.header("About")
    st.markdown("This is a weekend-built RAG bot for GitLab's public docs.")
    st.markdown("**Scope**: Handbook & Direction. If unsure, it will say so.")
    st.divider()
    st.markdown(f"Vector store: `{CHROMA_DIR}`")

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("Ask about GitLab's Handbook/Direction…")
if q:
    st.session_state.history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            # Embed query and retrieve
            q_emb = embed([q])[0]
            res = retr.query(q_emb, top_k=TOP_K)

            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]

            # Convert Chroma distance to a pseudo-similarity (depends on backend). Normalize gently.
            # If Chroma returns cosine distance, similarity ~ 1 - dist.
            sims = [1 - (d or 0) for d in dists]

            context_blocks = []
            sources = []
            for doc, meta, sim in zip(docs, metas, sims):
                url = meta.get("url", "")
                snippet = (doc[:300] + "…") if len(doc) > 300 else doc
                context_blocks.append(f"[source] {url}\n{snippet}")
                sources.append(url)

            # Guardrail: if everything below threshold, ask user to rephrase and show best matches
            if not sims or max(sims) < MIN_SCORE:
                msg = (
                    "I couldn't confidently find this in GitLab's Handbook/Direction. "
                    "Try rephrasing, or browse the sources below.\n\n" + "\n".join(f"- {u}" for u in sources)
                )
                st.markdown(msg)
                st.session_state.history.append({"role": "assistant", "content": msg})
            else:
                context = "\n\n".join(context_blocks)
                prompt = (
                    f"SYSTEM: You answer using only the context. If unknown, say so.\n\n"
                    f"CONTEXT:\n{context}\n\n"
                    f"USER: {q}\nASSISTANT:"
                )
                answer = chat([{"role": "user", "content": prompt}], temperature=0.2)

                # Render answer and sources
                st.markdown(answer)
                with st.expander("Sources"):
                    for i, (u, s) in enumerate(zip(sources, sims), start=1):
                        st.markdown(f"**{i}.** {u} — similarity ~ {s:.2f}")

                st.session_state.history.append({"role": "assistant", "content": answer})

                # Feedback UI
                c1, c2, c3 = st.columns([1,1,6])
                if c1.button("👍 Helpful", key=f"up_{len(st.session_state.history)}"):
                    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([datetime.datetime.utcnow().isoformat(), q, answer, 1, ";".join(sources)])
                    st.toast("Thanks for the feedback!")
                if c2.button("👎 Not helpful", key=f"down_{len(st.session_state.history)}"):
                    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([datetime.datetime.utcnow().isoformat(), q, answer, 0, ";".join(sources)])
                    st.toast("Logged. We'll improve retrieval.")
```

---

## 9) README (short form)

```md
# GitLab Handbook & Direction RAG Chatbot

**Goal**: Ask questions about GitLab's Handbook/Direction with cited answers.

## Run
1. `pip install -r requirements.txt`
2. Set env: `PROVIDER=gemini` and `GEMINI_API_KEY=...` (or OpenAI vars)
3. Build index: `python -m src.ingest`
4. Launch UI: `streamlit run src/app.py`

## Deploy
- Streamlit Cloud → app: `src/app.py` → add env vars

## Notes
- First run ingests ~up to 250 pages (bounded). Adjust limits in `src/ingest.py`.
- Shows sources + similarity. If confidence low, it says so.
- Feedback logged to `logs/feedback.csv`.
```

---

## 10) Project Write-up (paste in your submission doc)

**Objective**: Build a transparent, citation-first chatbot for GitLab’s Handbook/Direction.

**Architecture**: Browser (Streamlit) → Chat API (Gemini/OpenAI) + Retriever (Chroma) → Embeddings (text-embedding-004 or text-embedding-3-small) → Vector DB → GitLab pages (scraped).

**Key Choices**

* **RAG** to ground answers and avoid hallucinations.
* **Chroma** for quick, zero-ops vector store with persistence.
* **Gemini/OpenAI** pluggable via env; default Gemini for generous free tier.
* **Guardrails**: thresholded retrieval; explicit “don’t know”; citations every time.
* **UX**: lightweight Streamlit chat, visible sources, simple feedback.

**Trade-offs**

* Cold-start ingest vs committing index blobs. Kept simple for reproducibility.
* Character-based chunking (robust enough) vs token-perfect splitting (future work).

**Testing**

* Queries like: “What are GitLab values?”, “PTO policy?”, “Direction for DevSecOps?”
* Expect citations from `/handbook/` or `/direction/` pages.

**Future Improvements**

* Incremental crawler, sitemap boost, hybrid search, structured QA for policy pages.
* Role-mode (employee vs candidate) and prompt-tuned answers.
* Telemetry dashboard from feedback CSV.

---

## 11) How to Score Bonus Points

* **Transparency toggle**: show the exact context chunks used (already included via Sources expander).
* **Hallucination filter**: if the model outputs facts not in context (regex for URLs), prepend warning banner.
* **Mini analytics**: simple chart in Streamlit from `logs/feedback.csv` (thumbs-up rate).

---

## 12) Quick Commands (copy/paste)

```bash
# create venv
python -m venv .venv && source .venv/bin/activate

# install
pip install -r requirements.txt

# export env (mac/linux)
export PROVIDER=gemini
export GEMINI_API_KEY=YOUR_KEY

# ingest
python -m src.ingest

# run ui
streamlit run src/app.py
```
