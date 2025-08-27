import os, re, time, hashlib
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from .config import USER_AGENT, CACHE_DIR, ALLOWED_NETLOC, REQUEST_TIMEOUT, CRAWL_DELAY

os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(url: str):
    h = hashlib.sha256(url.encode()).hexdigest()[:20]
    return os.path.join(CACHE_DIR, f"{h}.html")

def fetch(url: str, timeout=None) -> str:
    """Fetch with simple on-disk cache to speed up dev."""
    cp = _cache_path(url)
    if os.path.exists(cp):
        with open(cp, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if timeout is None:
        timeout = REQUEST_TIMEOUT
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    html = r.text
    with open(cp, "w", encoding="utf-8") as f:
        f.write(html)
    time.sleep(CRAWL_DELAY)  # be polite
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

def chunk_text_simple(text: str, size=1200, overlap=150):
    """
    Simple fixed chunking function - backup method.
    Fixed version that prevents infinite loops.
    """
    if not text or size <= 0:
        return []
    
    chunks = []
    i = 0
    n = len(text)
    
    while i < n:
        # Get the end position for this chunk
        j = min(i + size, n)
        chunk = text[i:j].strip()
        
        # Only add non-empty chunks
        if chunk:
            chunks.append(chunk)
        
        # Calculate next starting position
        next_i = j - overlap
        
        # Prevent infinite loop: ensure we always move forward
        if next_i <= i:
            next_i = i + 1
        
        i = next_i
    
    return chunks


def chunk_text(text: str, size=1200, overlap=150):
    """
    Robust text chunking using LangChain text splitter.
    Falls back to simple method if LangChain is not available.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        # Use LangChain's robust text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        return [chunk.strip() for chunk in chunks if chunk.strip()]
        
    except ImportError:
        # Fallback to simple method if LangChain is not available
        print("Warning: LangChain not available, using simple chunking method")
        return chunk_text_simple(text, size, overlap)
    except Exception as e:
        # Fallback to simple method if LangChain fails
        print(f"Warning: LangChain chunking failed ({e}), using simple chunking method")
        return chunk_text_simple(text, size, overlap)
