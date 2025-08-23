# GitLab Handbook & Direction RAG Chatbot

A lean, end-to-end Retrieval-Augmented Generation (RAG) chatbot that answers questions from **GitLab's Handbook** and **Direction** pages. Built for fast weekend execution: scrape → chunk → embed → vector DB (Chroma) → LLM (Gemini or OpenAI) → Streamlit UI → Deploy.

## 🚀 Features

- **Core**: RAG over Handbook/Direction, citations per answer, follow-up friendly chat
- **Transparency**: Each answer shows sources (URL + snippet). If low confidence, the bot says so and links where to look
- **Guardrails**: Refuses to answer outside scope; fallback when retrieval is weak
- **Product thinking**: Feedback buttons (👍/👎) logged to CSV for iterative improvement

## 📁 Project Structure

```
.gitignore
README.md
requirements.txt
env.example             # Environment variables template
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

## 🛠️ Setup

### Prerequisites

- Python 3.10+
- API key for either Gemini or OpenAI

### Installation

1. **Clone and navigate to the project**
   ```bash
   git clone <your-repo-url>
   cd gitlab_handbook_rag_chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Copy the example environment file:
   ```bash
   cp env.example .env
   ```
   
   Edit `.env` and add your API key:
   ```bash
   # Choose one provider. "gemini" recommended (free tier generous)
   PROVIDER=gemini   # or openai
   
   # If PROVIDER=gemini
   GEMINI_API_KEY=your_actual_api_key_here
   
   # If PROVIDER=openai
   OPENAI_API_KEY=your_actual_api_key_here
   ```

### Getting API Keys

**Gemini (Recommended - Free tier with 1,500 calls/day):**
1. Go to [Google AI Studio](https://aistudio.google.com)
2. Create a new API key
3. Copy the key to your `.env` file

**OpenAI (Alternative):**
1. Go to [OpenAI Platform](https://platform.openai.com)
2. Create an API key
3. Copy the key to your `.env` file

## 🏃‍♂️ Running Locally

1. **Build the vector index** (scrape + embed)
   ```bash
   python -m src.ingest
   ```
   *This will crawl GitLab's Handbook and Direction pages and create embeddings. Takes 5-15 minutes on first run.*

2. **Launch the UI**
   ```bash
   streamlit run src/app.py
   ```

3. **Open your browser** to `http://localhost:8501`

## 🌐 Deployment

### Streamlit Community Cloud (Recommended)

1. Push this repo to GitHub (public or private with Streamlit access)
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) → New app
3. Select your repo/branch → set main file path to `src/app.py`
4. Add environment variables in the app's settings:
   - `PROVIDER=gemini`
   - `GEMINI_API_KEY=your_key_here`
5. Deploy! First run will ingest; subsequent runs reuse the persisted `data/chroma`

### Alternative Deployment Options

- **Hugging Face Spaces**: Upload files and set `app.py` as main file
- **Vercel**: Works with Python runtime
- **Railway/Render**: Standard Python app deployment

## 💡 Usage Examples

Try asking questions like:
- "What are GitLab's values?"
- "What is GitLab's PTO policy?"
- "What is GitLab's direction for DevSecOps?"
- "How does GitLab handle remote work?"
- "What are the engineering principles at GitLab?"

## 🔧 Configuration

You can tune the system via environment variables:

```bash
CHUNK_SIZE=1200      # characters per chunk
CHUNK_OVERLAP=150    # overlap between chunks
TOP_K=4              # number of chunks to retrieve
MIN_SCORE=0.78       # minimum similarity threshold (0-1)
```

## 🚧 Limitations

- Cold-start ingest takes time on first run (scraping + embeddings)
- Scope restricted to `/handbook/` and `/direction/` pages only
- Maximum 250 pages crawled to keep it bounded
- Not a full GitLab universe; focused on public documentation

## 🔮 Future Improvements

- Periodic background refresh via GitHub Action (nightly re-ingest)
- Add hybrid retrieval (BM25 + dense) for edge cases
- Inline source highlighting; semantic answer grading for guardrails
- Auth + role hints for employee vs candidate personas
- Analytics dashboard from feedback CSV

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally
5. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🆘 Troubleshooting

**Common Issues:**

1. **"GEMINI_API_KEY not set" error**
   - Make sure you've created a `.env` file with your API key
   - Check that the key is valid and has quota remaining

2. **Slow first run**
   - The initial ingestion crawls and embeds 250+ pages, this is normal
   - Subsequent runs are much faster as they reuse the vector database

3. **No results found**
   - Try rephrasing your question
   - The bot only knows about GitLab's Handbook and Direction pages
   - Check if your question is within scope

4. **Import errors**
   - Make sure you've installed all requirements: `pip install -r requirements.txt`
   - Check that you're using Python 3.10+

For more help, please open an issue on GitHub.
