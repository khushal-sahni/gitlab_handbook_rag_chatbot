"""
GitLab Documentation Ingestion Pipeline (Production Version)

This uses the new production-grade package structure.
Run with: python -m src.ingest
"""

import os
import sys
import queue
import time
import traceback
from pathlib import Path
from memory_profiler import profile

# SQLite compatibility fix for ChromaDB on deployment platforms
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from new production-grade package
from gitlab_rag_chatbot.config.settings import settings
from gitlab_rag_chatbot.core.embeddings.providers import get_embedding_function
from gitlab_rag_chatbot.core.retrieval.vector_store import DocumentRetriever
from gitlab_rag_chatbot.core.ingestion.progress_tracker import IngestionProgressTracker
from gitlab_rag_chatbot.utils.web_operations import WebContentFetcher, GitLabLinkExtractor, HTMLContentExtractor
from gitlab_rag_chatbot.utils.text_processing import DocumentTextSplitter
from gitlab_rag_chatbot.utils.memory_monitoring import SystemMemoryMonitor
from gitlab_rag_chatbot.utils.logging_setup import setup_application_logging

# Setup comprehensive logging
logger = setup_application_logging(
    log_level="INFO",
    enable_console_logging=True,
    enable_file_logging=True
)

# Initialize components
memory_monitor = SystemMemoryMonitor()
web_fetcher = WebContentFetcher()
link_extractor = GitLabLinkExtractor()
content_extractor = HTMLContentExtractor()
text_splitter = DocumentTextSplitter()


@profile
def build_index():
    """
    Execute the complete document ingestion pipeline with production-grade features.
    """
    logger.info("🚀 Starting GitLab Documentation Ingestion Pipeline")
    logger.info("=" * 60)
    
    # Log configuration
    config_summary = settings.get_configuration_summary()
    logger.info("Configuration Summary:")
    for key, value in config_summary.items():
        logger.info(f"  {key}: {value}")
    
    # Initialize progress tracker
    progress_tracker = IngestionProgressTracker()
    
    # Set configuration snapshot for validation
    config_snapshot = {
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_batch_size": settings.embedding_batch_size,
        "max_pages_to_crawl": settings.max_pages_to_crawl,
        "embedding_provider": settings.embedding_provider
    }
    
    # Check if resuming from checkpoint
    if settings.resume_from_checkpoint:
        if not progress_tracker.validate_configuration_consistency(config_snapshot):
            logger.warning("Configuration mismatch detected. Starting fresh to avoid inconsistencies.")
            progress_tracker.clear_checkpoint()
        else:
            stats = progress_tracker.get_progress_statistics()
            logger.info(f"Resuming from checkpoint: {stats}")
    else:
        logger.info("Starting fresh ingestion")
        progress_tracker.clear_checkpoint()
    
    progress_tracker.set_configuration_snapshot(config_snapshot)
    
    try:
        with memory_monitor.monitor_operation("pipeline_initialization"):
            # Initialize components
            logger.info("🤖 Initializing embedding provider...")
            embedding_function = get_embedding_function()
            
            logger.info("🗄️ Initializing document retriever...")
            document_retriever = DocumentRetriever()
            
            # Ensure directories exist
            settings.ensure_directories_exist()
        
        # Initialize crawling state
        discovered_urls = progress_tracker.get_discovered_urls()
        url_queue = queue.Queue()
        
        if not discovered_urls:
            # Starting fresh - add seed URLs
            for url in settings.seed_urls:
                url_queue.put(url)
                discovered_urls.add(url)
        else:
            # Restore from checkpoint
            for url in progress_tracker.get_pending_urls_queue():
                url_queue.put(url)
            logger.info(f"Restored {url_queue.qsize()} URLs from checkpoint")
        
        pages_processed = len(progress_tracker.checkpoint_state["processed_urls"])
        total_chunks_processed = progress_tracker.checkpoint_state["total_chunks_processed"]
        
        logger.info(f"📊 Starting state: {pages_processed} pages, {total_chunks_processed} chunks processed")
        
        # Main processing loop
        while not url_queue.empty() and pages_processed < settings.max_pages_to_crawl:
            current_url = url_queue.get()
            
            # Skip if already processed
            if progress_tracker.should_skip_url(current_url):
                logger.debug(f"Skipping already processed URL: {current_url}")
                continue

            logger.info(f"📄 Processing page {pages_processed + 1}/{settings.max_pages_to_crawl}: {current_url}")
            
            processing_start_time = time.time()
            
            try:
                with memory_monitor.monitor_operation(f"page_processing_{pages_processed}"):
                    # Fetch and process content
                    logger.info("🌐 Fetching HTML content...")
                    html_content = web_fetcher.fetch_content(current_url)
                    
                    # Extract new links
                    logger.info("🔗 Extracting links...")
                    new_links = link_extractor.extract_documentation_links(current_url, html_content)
                    
                    new_links_count = 0
                    for link in new_links:
                        if link not in discovered_urls:
                            discovered_urls.add(link)
                            url_queue.put(link)
                            new_links_count += 1
                    
                    logger.info(f"Found {new_links_count} new links, queue size: {url_queue.qsize()}")
                    
                    # Convert HTML to text
                    logger.info("📝 Converting HTML to text...")
                    text_content = content_extractor.extract_text_content(html_content)
                    
                    # Split into chunks
                    logger.info("✂️ Splitting text into chunks...")
                    document_chunks = text_splitter.split_text_into_chunks(text_content)
                    
                    # Get chunking statistics
                    chunk_stats = text_splitter.get_chunk_statistics(document_chunks)
                    logger.info(f"Created {chunk_stats['total_chunks']} chunks "
                               f"(avg: {chunk_stats['average_chunk_length']:.0f} chars)")
                    
                    # Process chunks in batches
                    document_ids = []
                    document_texts = []
                    document_metadata = []
                    page_chunks_processed = 0
                    
                    for chunk_index, chunk_text in enumerate(document_chunks):
                        document_ids.append(f"{pages_processed}-{chunk_index}")
                        document_texts.append(chunk_text)
                        document_metadata.append({"url": current_url})
                        
                        # Process batch when it reaches the configured size
                        if len(document_texts) >= settings.embedding_batch_size:
                            batch_number = (page_chunks_processed // settings.embedding_batch_size) + 1
                            
                            with memory_monitor.monitor_operation(f"embedding_batch_{batch_number}"):
                                logger.info(f"🧠 Generating embeddings for batch {batch_number} "
                                           f"({len(document_texts)} chunks)...")
                                
                                # Generate embeddings
                                document_embeddings = embedding_function(document_texts)
                                
                                # Store in vector database
                                logger.info("💾 Storing documents in vector database...")
                                document_retriever.add_documents(
                                    document_ids=document_ids,
                                    document_texts=document_texts,
                                    document_metadata=document_metadata,
                                    document_embeddings=document_embeddings
                                )
                                
                                batch_size = len(document_texts)
                                total_chunks_processed += batch_size
                                page_chunks_processed += batch_size
                                
                                logger.info(f"✅ Processed batch: {batch_size} chunks "
                                           f"(total: {total_chunks_processed})")
                                
                                # Clear batch data
                                document_ids = []
                                document_texts = []
                                document_metadata = []
                                
                                # Save checkpoint periodically
                                if total_chunks_processed % settings.checkpoint_interval == 0:
                                    progress_tracker.update_progress(
                                        discovered_urls=discovered_urls,
                                        pending_urls_queue=list(url_queue.queue)
                                    )
                                    progress_tracker.save_checkpoint()
                                    logger.info(f"💾 Checkpoint saved at {total_chunks_processed} chunks")
                    
                    # Process remaining chunks
                    if document_texts:
                        with memory_monitor.monitor_operation("final_embedding_batch"):
                            logger.info(f"🧠 Processing final batch ({len(document_texts)} chunks)...")
                            
                            document_embeddings = embedding_function(document_texts)
                            
                            document_retriever.add_documents(
                                document_ids=document_ids,
                                document_texts=document_texts,
                                document_metadata=document_metadata,
                                document_embeddings=document_embeddings
                            )
                            
                            batch_size = len(document_texts)
                            total_chunks_processed += batch_size
                            page_chunks_processed += batch_size
                            
                            logger.info(f"✅ Final batch processed: {batch_size} chunks")
                    
                    # Update progress
                    processing_duration = time.time() - processing_start_time
                    
                    progress_tracker.update_progress(
                        processed_url=current_url,
                        chunks_processed_count=page_chunks_processed,
                        discovered_urls=discovered_urls,
                        pending_urls_queue=list(url_queue.queue),
                        processing_duration_seconds=processing_duration
                    )
                    
                    pages_processed += 1
                    
                    # Save checkpoint after each page
                    progress_tracker.save_checkpoint()
                    
                    logger.info(f"✅ Page completed: {page_chunks_processed} chunks processed "
                               f"in {processing_duration:.1f}s")
                    
            except Exception as e:
                logger.error(f"❌ Error processing {current_url}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                # Save checkpoint before continuing
                progress_tracker.update_progress(
                    discovered_urls=discovered_urls,
                    pending_urls_queue=list(url_queue.queue)
                )
                progress_tracker.save_checkpoint()
                continue
        
        # Final statistics
        final_stats = progress_tracker.get_progress_statistics()
        collection_stats = document_retriever.get_collection_stats()
        
        logger.info("🎉 Ingestion Pipeline Completed Successfully!")
        logger.info("=" * 60)
        logger.info(f"📊 Final Statistics:")
        logger.info(f"  Pages processed: {final_stats['pages_processed']}")
        logger.info(f"  Total chunks: {final_stats['total_chunks_processed']}")
        logger.info(f"  Documents in vector store: {collection_stats['total_documents']}")
        logger.info(f"  Average chunks per page: {final_stats.get('average_chunks_per_page', 0):.1f}")
        logger.info(f"  Vector store path: {collection_stats['vector_store_path']}")
        
        # Clear checkpoint on successful completion
        progress_tracker.clear_checkpoint()
        logger.info("✅ Checkpoint cleared after successful completion")
        
    except Exception as e:
        logger.error(f"❌ Critical error in ingestion pipeline: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Save checkpoint for recovery
        try:
            progress_tracker.save_checkpoint()
            logger.info("💾 Checkpoint saved for recovery")
        except Exception as checkpoint_error:
            logger.error(f"Failed to save recovery checkpoint: {checkpoint_error}")
        
        raise


if __name__ == "__main__":
    build_index()
