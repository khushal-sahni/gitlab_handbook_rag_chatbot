# GitLab RAG Chatbot - Production Package

A production-grade Retrieval-Augmented Generation (RAG) chatbot for GitLab's public documentation, built with enterprise-level code quality, comprehensive error handling, and professional software engineering practices.

## 🏗️ Architecture Overview

This package follows a clean, modular architecture with clear separation of concerns:

```
gitlab_rag_chatbot/
├── 📦 config/              # Configuration Management
│   ├── constants.py        # Application constants & defaults
│   └── settings.py         # Environment-based configuration
├── 🧠 core/               # Core Business Logic
│   ├── embeddings/        # AI embedding providers (Gemini, OpenAI)
│   ├── ingestion/         # Document processing pipeline
│   └── retrieval/         # Vector storage & semantic search
├── 🛠️ utils/              # Utility Functions
│   ├── logging_setup.py   # Centralized logging configuration
│   ├── memory_monitoring.py # System resource monitoring
│   ├── text_processing.py # Document chunking & processing
│   └── web_operations.py  # Web scraping & content extraction
└── 🌐 web/               # Web Interface
    └── feedback_collector.py # User feedback management
```

## 🚀 Key Features

### Production-Grade Code Quality
- **Type Safety**: Comprehensive type hints throughout
- **Error Handling**: Robust exception management with retry logic
- **Documentation**: Extensive docstrings and inline comments
- **Logging**: Structured, configurable logging system
- **Memory Management**: Advanced memory monitoring and cleanup
- **Configuration**: Environment-based settings with validation

### Advanced RAG Capabilities
- **Multi-Provider Embeddings**: Support for Gemini and OpenAI
- **Intelligent Chunking**: LangChain integration with fallback methods
- **Semantic Search**: ChromaDB-powered vector similarity search
- **Progress Tracking**: Robust checkpointing for fault tolerance
- **Rate Limiting**: Intelligent API request management

### Enterprise Features
- **Scalable Architecture**: Modular design for easy extension
- **Monitoring**: Comprehensive memory and performance tracking
- **Feedback Loop**: User feedback collection for continuous improvement
- **Caching**: Intelligent web content caching
- **Retry Logic**: Exponential backoff for API reliability

## 📋 Usage Examples

### Basic Usage

```python
from gitlab_rag_chatbot.core.embeddings import EmbeddingProvider
from gitlab_rag_chatbot.core.retrieval import DocumentRetriever
from gitlab_rag_chatbot.config.settings import settings

# Initialize components
embedding_function = EmbeddingProvider.create_embedding_function()
retriever = DocumentRetriever()

# Generate embeddings and search
query_embedding = embedding_function(["How to configure CI/CD?"])[0]
results = retriever.search_similar_documents(query_embedding)
```

### Advanced Configuration

```python
from gitlab_rag_chatbot.config.settings import ApplicationSettings

# Custom configuration
settings = ApplicationSettings(env_file_path="/path/to/custom/.env")

# Get configuration summary
config_summary = settings.get_configuration_summary()
print(config_summary)
```

### Memory Monitoring

```python
from gitlab_rag_chatbot.utils.memory_monitoring import SystemMemoryMonitor

monitor = SystemMemoryMonitor()

# Monitor an operation
with monitor.monitor_operation("document_processing"):
    # Your memory-intensive operation here
    pass

# Get memory statistics
stats = monitor.get_current_memory_usage()
print(f"Memory usage: {stats['resident_set_size_mb']:.1f}MB")
```

### Text Processing

```python
from gitlab_rag_chatbot.utils.text_processing import DocumentTextSplitter

splitter = DocumentTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_text_into_chunks(document_text)

# Get chunking statistics
stats = splitter.get_chunk_statistics(chunks)
print(f"Created {stats['total_chunks']} chunks")
```

## 🔧 Configuration

All configuration is managed through environment variables with sensible defaults:

### Core Settings
- `PROVIDER`: Embedding provider ("gemini" or "openai")
- `GEMINI_API_KEY`: Google Gemini API key
- `OPENAI_API_KEY`: OpenAI API key

### Document Processing
- `CHUNK_SIZE`: Text chunk size (default: 1200)
- `CHUNK_OVERLAP`: Chunk overlap (default: 150)
- `MAX_PAGES`: Maximum pages to crawl (default: 50)

### Performance Tuning
- `EMBEDDING_BATCH_SIZE`: Embedding batch size (default: 64)
- `REQUEST_TIMEOUT`: HTTP request timeout (default: 30)
- `CRAWL_DELAY`: Delay between requests (default: 0.1)

### API Reliability
- `MAX_RETRIES`: Maximum API retries (default: 5)
- `RETRY_BASE_DELAY`: Base retry delay (default: 1.0)
- `API_TIMEOUT`: API operation timeout (default: 120)

See `config/settings.py` for complete configuration options.

## 🧪 Testing & Development

### Memory Profiling
```python
from gitlab_rag_chatbot.utils.memory_monitoring import system_memory_monitor

# Log memory usage
memory_stats = system_memory_monitor.log_memory_usage("operation_name")

# Perform cleanup if needed
if system_memory_monitor.is_memory_usage_high():
    system_memory_monitor.perform_memory_cleanup()
```

### Logging Configuration
```python
from gitlab_rag_chatbot.utils.logging_setup import setup_application_logging

# Setup comprehensive logging
logger = setup_application_logging(
    log_level="DEBUG",
    enable_console_logging=True,
    enable_file_logging=True
)
```

## 🔄 Migration from Old Structure

If you're migrating from the old flat structure, use the migration script:

```bash
python scripts/migrate_to_new_structure.py
```

This will:
- Backup your existing data
- Preserve all configurations
- Create the new package structure
- Provide a detailed migration guide

## 🎯 Best Practices

### Error Handling
```python
from gitlab_rag_chatbot.core.embeddings.retry_handler import with_retry

@with_retry(max_retries=3)
def api_operation():
    # Your API call here
    pass
```

### Resource Management
```python
from gitlab_rag_chatbot.utils.memory_monitoring import SystemMemoryMonitor

monitor = SystemMemoryMonitor()

# Always monitor memory-intensive operations
with monitor.monitor_operation("embedding_generation"):
    embeddings = embedding_function(large_text_batch)
```

### Configuration Management
```python
from gitlab_rag_chatbot.config.settings import settings

# Always use centralized settings
chunk_size = settings.chunk_size  # ✅ Good
# chunk_size = 1200  # ❌ Avoid hardcoding
```

## 📊 Performance Considerations

- **Batch Processing**: Use appropriate batch sizes for embeddings
- **Memory Management**: Monitor memory usage during large operations
- **Caching**: Leverage web content caching for development
- **Rate Limiting**: Respect API rate limits with built-in delays
- **Checkpointing**: Use progress tracking for long-running operations

## 🔍 Monitoring & Observability

The package includes comprehensive monitoring capabilities:

- **Memory Usage**: Real-time memory tracking and alerts
- **API Performance**: Request timing and retry statistics
- **Processing Metrics**: Document processing and chunking stats
- **User Feedback**: Structured feedback collection and analysis

## 🤝 Contributing

This codebase follows enterprise-level standards:

1. **Type Hints**: All functions must have complete type annotations
2. **Documentation**: Comprehensive docstrings required
3. **Error Handling**: Robust exception management
4. **Logging**: Appropriate logging levels and messages
5. **Testing**: Unit tests for all core functionality
6. **Performance**: Memory and performance considerations

## 📈 Extensibility

The modular architecture makes it easy to extend:

- **New Embedding Providers**: Implement `BaseEmbeddingProvider`
- **Custom Text Splitters**: Extend `DocumentTextSplitter`
- **Additional Monitoring**: Extend `SystemMemoryMonitor`
- **New Web Interfaces**: Add modules to `web/` package
- **Custom Retrievers**: Implement new retrieval strategies

---

*This package represents production-grade software engineering practices applied to RAG systems, ensuring reliability, maintainability, and scalability for enterprise use.*
