import os, sys, queue, gc, logging, time, traceback
from urllib.parse import urlparse
import psutil
import tracemalloc
from .config import (
    SEED_URLS, CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_DIR, MIN_SCORE, DATA_DIR, LOG_DIR,
    MAX_PAGES, EMBEDDING_BATCH_SIZE, CHECKPOINT_INTERVAL, RESUME_FROM_CHECKPOINT
)
from .utils import fetch, extract_links, html_to_text, chunk_text
from .model_providers.embeddings import get_embedding_fn
from .retriever import Retriever
from .checkpoint import IngestionCheckpoint
from memory_profiler import profile

# Setup comprehensive logging
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'ingest_debug.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration is now loaded from environment variables in config.py


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        'percent': process.memory_percent(),
        'available_mb': psutil.virtual_memory().available / 1024 / 1024
    }


def log_memory_usage(step_name):
    """Log current memory usage with step context"""
    mem = get_memory_usage()
    logger.info(f"MEMORY [{step_name}] - RSS: {mem['rss_mb']:.1f}MB, VMS: {mem['vms_mb']:.1f}MB, "
                f"Percent: {mem['percent']:.1f}%, Available: {mem['available_mb']:.1f}MB")
    
    # Alert if memory usage is high
    if mem['percent'] > 80:
        logger.warning(f"HIGH MEMORY USAGE: {mem['percent']:.1f}% - Available: {mem['available_mb']:.1f}MB")
    
    return mem


def cleanup_memory():
    """Force garbage collection and log memory cleanup"""
    before = get_memory_usage()
    gc.collect()
    after = get_memory_usage()
    freed_mb = before['rss_mb'] - after['rss_mb']
    logger.info(f"MEMORY CLEANUP - Freed: {freed_mb:.1f}MB, "
                f"Before: {before['rss_mb']:.1f}MB, After: {after['rss_mb']:.1f}MB")


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


@profile
def build_index():
    logger.info("Starting build_index process")
    logger.info(f"Configuration: MAX_PAGES={MAX_PAGES}, BATCH_SIZE={EMBEDDING_BATCH_SIZE}, "
                f"CHUNK_SIZE={CHUNK_SIZE}, CHECKPOINT_INTERVAL={CHECKPOINT_INTERVAL}")
    
    # Initialize checkpoint system
    checkpoint = IngestionCheckpoint()
    
    # Set config snapshot for validation
    config_snapshot = {
        "CHUNK_SIZE": CHUNK_SIZE,
        "CHUNK_OVERLAP": CHUNK_OVERLAP,
        "EMBEDDING_BATCH_SIZE": EMBEDDING_BATCH_SIZE,
        "MAX_PAGES": MAX_PAGES
    }
    
    # Check if resuming from checkpoint
    if RESUME_FROM_CHECKPOINT and checkpoint.load():
        if not checkpoint.validate_config(config_snapshot):
            logger.warning("Config mismatch detected. Starting fresh to avoid inconsistencies.")
            checkpoint.clear()
        else:
            logger.info("Resuming from checkpoint...")
            stats = checkpoint.get_stats()
            logger.info(f"Previous progress: {stats}")
    else:
        logger.info("Starting fresh ingestion")
        checkpoint.clear()
    
    checkpoint.set_config_snapshot(config_snapshot)
    
    # Start memory tracking
    tracemalloc.start()
    log_memory_usage("START")
    
    try:
        logger.info("Initializing embedding function")
        embed = get_embedding_fn()
        log_memory_usage("AFTER_EMBED_INIT")
        
        logger.info("Initializing Retriever (ChromaDB)")
        r = Retriever()
        log_memory_usage("AFTER_RETRIEVER_INIT")

        # Initialize or restore state
        seen = checkpoint.get_seen_urls()
        q = queue.Queue()
        
        # Add seed URLs or restore queue
        if not seen:
            for u in SEED_URLS:
                q.put(u)
                seen.add(u)
        else:
            # Restore queue from checkpoint
            for url in checkpoint.get_queue_urls():
                q.put(url)
            logger.info(f"Restored {q.qsize()} URLs from checkpoint queue")

        count_pages = len(checkpoint.state["processed_urls"])
        count_chunks = checkpoint.state["total_chunks_processed"]
        
        logger.info(f"Starting crawl with batch size: {EMBEDDING_BATCH_SIZE}, max pages: {MAX_PAGES}")
        logger.info(f"Resume state: {count_pages} pages, {count_chunks} chunks processed")

        while not q.empty() and count_pages < MAX_PAGES:
            url = q.get()
            
            # Skip if already processed
            if checkpoint.should_skip_url(url):
                logger.debug(f"Skipping already processed URL: {url}")
                continue
                
            logger.info(f"Processing page {count_pages + 1}/{MAX_PAGES}: {url}")
            log_memory_usage(f"BEFORE_PAGE_{count_pages}")
            
            try:
                logger.info(f"Fetching HTML for: {url}")
                html = fetch(url)
                html_size_mb = len(html) / 1024 / 1024
                logger.info(f"Fetched HTML size: {html_size_mb:.2f}MB")
                log_memory_usage("AFTER_FETCH")
                
            except Exception as e:
                logger.error(f"Fetch failed for {url}: {e}")
                continue

            # enqueue new links
            logger.info("Extracting links")
            links = extract_links(url, html)
            new_links = 0
            for l in links:
                if l not in seen:
                    seen.add(l)
                    q.put(l)
                    new_links += 1
            logger.info(f"Found {new_links} new links, queue size: {q.qsize()}")
            log_memory_usage("AFTER_LINK_EXTRACTION")

            # convert & chunk immediately
            logger.info("Converting HTML to text")
            text = html_to_text(html)
            text_size_mb = len(text) / 1024 / 1024
            logger.info(f"Converted text size: {text_size_mb:.2f}MB")
            log_memory_usage("AFTER_HTML_TO_TEXT")
            
            # Clear HTML from memory
            del html
            cleanup_memory()
            
            logger.info("Chunking text")
            chunks = chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            logger.info(f"Created {len(chunks)} chunks")
            log_memory_usage("AFTER_CHUNKING")
            
            # Clear text from memory
            del text
            cleanup_memory()

            # process chunks in batches
            ids, docs, metas = [], [], []
            batch_count = 0
            page_chunks_processed = 0
            
            for j, ch in enumerate(chunks):
                ids.append(f"{count_pages}-{j}")
                docs.append(ch)
                metas.append({"url": url})

                if len(docs) >= EMBEDDING_BATCH_SIZE:
                    batch_count += 1
                    logger.info(f"Processing embedding batch {batch_count} with {len(docs)} chunks")
                    log_memory_usage(f"BEFORE_EMBED_BATCH_{batch_count}")
                    
                    try:
                        embs = embed(docs)
                        log_memory_usage(f"AFTER_EMBED_BATCH_{batch_count}")
                        
                        logger.info(f"Adding batch to ChromaDB")
                        r.add(ids=ids, docs=docs, metas=metas, embs=embs)
                        log_memory_usage(f"AFTER_CHROMA_ADD_BATCH_{batch_count}")
                        
                        batch_size = len(docs)
                        count_chunks += batch_size
                        page_chunks_processed += batch_size
                        logger.info(f"Indexed {count_chunks} chunks so far")
                        
                        # Clear batch data from memory
                        del embs, ids, docs, metas
                        ids, docs, metas = [], [], []
                        cleanup_memory()
                        
                        # Save checkpoint periodically
                        if count_chunks % CHECKPOINT_INTERVAL == 0:
                            checkpoint.update_progress(
                                chunks_count=0,  # Already updated count_chunks
                                seen_urls=seen,
                                queue_urls=list(q.queue)
                            )
                            checkpoint.save()
                            logger.info(f"Checkpoint saved at {count_chunks} chunks")
                        
                    except Exception as e:
                        logger.error(f"Error processing batch {batch_count}: {e}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        # Save checkpoint before failing
                        checkpoint.update_progress(
                            seen_urls=seen,
                            queue_urls=list(q.queue)
                        )
                        checkpoint.save()
                        raise

            # flush remainder for this page
            if docs:
                batch_count += 1
                logger.info(f"Processing final batch {batch_count} with {len(docs)} chunks")
                log_memory_usage(f"BEFORE_FINAL_EMBED_BATCH")
                
                try:
                    embs = embed(docs)
                    log_memory_usage(f"AFTER_FINAL_EMBED_BATCH")
                    
                    logger.info(f"Adding final batch to ChromaDB")
                    r.add(ids=ids, docs=docs, metas=metas, embs=embs)
                    log_memory_usage(f"AFTER_FINAL_CHROMA_ADD")
                    
                    batch_size = len(docs)
                    count_chunks += batch_size
                    page_chunks_processed += batch_size
                    logger.info(f"Indexed {count_chunks} chunks so far")
                    
                    # Clear final batch data from memory
                    del embs, ids, docs, metas
                    cleanup_memory()
                    
                except Exception as e:
                    logger.error(f"Error processing final batch: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # Save checkpoint before failing
                    checkpoint.update_progress(
                        seen_urls=seen,
                        queue_urls=list(q.queue)
                    )
                    checkpoint.save()
                    raise

            # Clear chunks from memory
            del chunks
            cleanup_memory()
            
            # Update checkpoint with completed page
            checkpoint.update_progress(
                processed_url=url,
                chunks_count=page_chunks_processed,
                seen_urls=seen,
                queue_urls=list(q.queue)
            )
            
            count_pages += 1
            log_memory_usage(f"AFTER_PAGE_{count_pages}")
            
            # Save checkpoint after each page
            checkpoint.save()
            logger.info(f"Page {count_pages} completed: {page_chunks_processed} chunks processed")
            
            # Force cleanup after each page
            cleanup_memory()

        logger.info(f"Finished: {count_pages} pages, {count_chunks} chunks total")
        logger.info(f"Index built and persisted to: {CHROMA_DIR}")
        log_memory_usage("FINAL")
        
        # Clear checkpoint on successful completion
        checkpoint.clear()
        logger.info("Ingestion completed successfully. Checkpoint cleared.")
        
        # Get final memory snapshot
        current, peak = tracemalloc.get_traced_memory()
        logger.info(f"Memory tracing - Current: {current / 1024 / 1024:.1f}MB, Peak: {peak / 1024 / 1024:.1f}MB")
        
    except Exception as e:
        logger.error(f"Critical error in build_index: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        log_memory_usage("ERROR")
        
        # Save checkpoint on error for resume
        try:
            checkpoint.save()
            logger.info("Checkpoint saved for resume after error")
        except:
            logger.error("Failed to save checkpoint on error")
        
        raise
    finally:
        tracemalloc.stop()

if __name__ == "__main__":
    build_index()
