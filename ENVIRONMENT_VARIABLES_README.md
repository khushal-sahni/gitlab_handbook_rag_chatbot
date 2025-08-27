# Environment Variables Guide

This document explains all environment variables used in the GitLab Handbook RAG Chatbot, what they do, and the impact of changing them.

## 📋 Complete Environment Variables Reference

### 🔑 API Configuration

#### `PROVIDER` (gemini | openai)
- **What it does**: Selects which AI service to use for embeddings
- **Default**: `gemini`
- **Effect of changing**:
  - `gemini`: Uses Google's Gemini API (free tier: 15 requests/minute)
  - `openai`: Uses OpenAI's API (paid service, higher rate limits)
  - **Impact**: Different cost structures, rate limits, and embedding quality

#### `GEMINI_API_KEY` / `OPENAI_API_KEY`
- **What it does**: Authentication credentials for the chosen AI service
- **Effect**: Must match your `PROVIDER` choice, enables API access

---

### 📝 Text Processing Settings

#### `CHUNK_SIZE=1200`
- **What it does**: Maximum characters per text chunk before embedding
- **Effect of changing**:
  - **Increase (1500-2000)**: 
    - ✅ More context per chunk, better semantic understanding
    - ❌ Higher API costs, slower processing, may hit token limits
  - **Decrease (800-1000)**:
    - ✅ Faster processing, lower costs, more precise matching
    - ❌ Less context, may break up related information

#### `CHUNK_OVERLAP=150`
- **What it does**: Characters that overlap between adjacent chunks
- **Effect of changing**:
  - **Increase (200-300)**: 
    - ✅ Better continuity, less information loss at boundaries
    - ❌ More redundant data, higher processing costs
  - **Decrease (50-100)**:
    - ✅ Less redundancy, faster processing
    - ❌ Risk of losing context at chunk boundaries

#### `TOP_K=4`
- **What it does**: Number of most relevant chunks returned for each query
- **Effect of changing**:
  - **Increase (6-10)**: More comprehensive answers, higher chance of finding relevant info
  - **Decrease (2-3)**: Faster responses, more focused answers, less noise

#### `MIN_SCORE=0.78`
- **What it does**: Minimum similarity threshold (0-1) for including chunks in results
- **Effect of changing**:
  - **Increase (0.85-0.95)**: Only very relevant results, fewer false positives
  - **Decrease (0.65-0.75)**: More results included, higher chance of finding matches

---

### ⚡ Performance & Speed Settings

#### `MAX_PAGES=50`
- **What it does**: Maximum number of web pages to crawl and index
- **Effect of changing**:
  - **Increase (100-500)**: 
    - ✅ More comprehensive knowledge base
    - ❌ Much longer processing time, higher API costs
  - **Decrease (10-25)**: 
    - ✅ Faster initial setup, good for testing
    - ❌ Limited knowledge coverage

#### `EMBEDDING_BATCH_SIZE=64`
- **What it does**: Number of text chunks processed together in one API call
- **Effect of changing**:
  - **Increase (96-128)**: 
    - ✅ Better API efficiency, faster overall processing
    - ❌ Higher memory usage, risk of API timeouts
  - **Decrease (32-48)**: 
    - ✅ Lower memory usage, more stable for limited resources
    - ❌ More API calls, slower overall processing

#### `CRAWL_DELAY=0.1`
- **What it does**: Seconds to wait between web page requests (politeness)
- **Effect of changing**:
  - **Increase (0.3-1.0)**: 
    - ✅ More respectful to target servers, less likely to be blocked
    - ❌ Significantly slower crawling
  - **Decrease (0.05-0.0)**: 
    - ✅ Faster crawling
    - ❌ Risk of being rate-limited or blocked by servers

#### `REQUEST_TIMEOUT=30`
- **What it does**: Seconds to wait for web page responses before giving up
- **Effect of changing**:
  - **Increase (45-60)**: Better success with slow servers, fewer failed requests
  - **Decrease (15-20)**: Faster failure detection, but may miss slow-loading pages

---

### 🔄 Reliability & Error Handling

#### `MAX_RETRIES=5`
- **What it does**: How many times to retry failed API calls before giving up
- **Effect of changing**:
  - **Increase (7-10)**: 
    - ✅ Higher success rate with unreliable connections
    - ❌ Much longer delays when APIs are truly down
  - **Decrease (2-3)**: 
    - ✅ Faster failure detection
    - ❌ More likely to fail on temporary issues

#### `RETRY_BASE_DELAY=1.0`
- **What it does**: Starting delay (seconds) for exponential backoff retries
- **Effect of changing**:
  - **Increase (2.0-5.0)**: Longer waits, better for rate-limited APIs
  - **Decrease (0.5)**: Faster retries, but may hit rate limits again quickly

#### `RETRY_MAX_DELAY=60.0`
- **What it does**: Maximum seconds to wait between retry attempts
- **Effect of changing**:
  - **Increase (120-300)**: Will wait longer for APIs to recover
  - **Decrease (30)**: Gives up faster on persistent issues

#### `API_TIMEOUT=120`
- **What it does**: Maximum seconds to wait for embedding API responses
- **Effect of changing**:
  - **Increase (180-300)**: Better for large batches, slower connections
  - **Decrease (60-90)**: Faster timeout detection, but may fail on large batches

---

### 💾 Progress Management

#### `CHECKPOINT_INTERVAL=100`
- **What it does**: Save progress every N processed chunks
- **Effect of changing**:
  - **Decrease (50)**: 
    - ✅ More frequent saves, less work lost on failures
    - ❌ Slightly slower due to more disk I/O
  - **Increase (200-500)**: 
    - ✅ Less disk I/O overhead
    - ❌ More work lost if process fails

#### `RESUME_FROM_CHECKPOINT=true`
- **What it does**: Whether to automatically continue from saved progress
- **Effect of changing**:
  - **`false`**: Always starts fresh, ignores previous progress
  - **`true`**: Resumes where it left off (recommended for production)

---

## 🎯 Recommended Settings by Use Case

### 🧪 Testing/Development
```bash
MAX_PAGES=5
EMBEDDING_BATCH_SIZE=16
CHECKPOINT_INTERVAL=10
CRAWL_DELAY=0.5
```
**Use when**: Testing changes, debugging, or learning the system

### ⚡ Fast Production
```bash
MAX_PAGES=100
EMBEDDING_BATCH_SIZE=96
CHECKPOINT_INTERVAL=200
CRAWL_DELAY=0.05
MAX_RETRIES=3
```
**Use when**: You need results quickly and have reliable internet/APIs

### 🛡️ Reliable/Conservative
```bash
MAX_PAGES=50
EMBEDDING_BATCH_SIZE=32
CHECKPOINT_INTERVAL=50
CRAWL_DELAY=0.3
MAX_RETRIES=7
RETRY_MAX_DELAY=120
```
**Use when**: Stability is more important than speed, unreliable connection

### 💰 Cost-Optimized
```bash
CHUNK_SIZE=1000
EMBEDDING_BATCH_SIZE=128
MAX_PAGES=25
CHUNK_OVERLAP=100
```
**Use when**: Minimizing API costs is the priority

---

## 🚀 Quick Start

1. Copy `env.example` to `.env`
2. Add your API key for your chosen provider
3. Adjust settings based on your use case above
4. Run the ingestion: `python -m src.ingest`

## 🔧 Troubleshooting

### Process gets killed
- **Reduce**: `EMBEDDING_BATCH_SIZE`, `MAX_PAGES`
- **Increase**: `CHECKPOINT_INTERVAL` for more frequent saves

### API timeouts
- **Increase**: `API_TIMEOUT`, `MAX_RETRIES`, `RETRY_MAX_DELAY`
- **Decrease**: `EMBEDDING_BATCH_SIZE`

### Too slow
- **Increase**: `EMBEDDING_BATCH_SIZE`, `MAX_PAGES`
- **Decrease**: `CRAWL_DELAY`, `CHECKPOINT_INTERVAL`

### Rate limited
- **Increase**: `CRAWL_DELAY`, `RETRY_BASE_DELAY`
- **Decrease**: `EMBEDDING_BATCH_SIZE`

### Poor answer quality
- **Increase**: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`
- **Decrease**: `MIN_SCORE`

---

**💡 Pro Tip**: Start with the "Testing/Development" settings first, then gradually optimize based on your specific needs and constraints!
