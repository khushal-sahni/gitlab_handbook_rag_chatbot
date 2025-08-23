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
