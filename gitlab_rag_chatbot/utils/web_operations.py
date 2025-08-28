"""
Web content fetching and HTML processing utilities.

This module provides robust web scraping capabilities with caching,
rate limiting, and intelligent content extraction for GitLab documentation.
"""

import os
import re
import time
import hashlib
import logging
from typing import Set, Optional
from urllib.parse import urlparse, urljoin
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from ..config.settings import settings
from ..config.constants import EXCLUDED_FILE_EXTENSIONS

logger = logging.getLogger(__name__)


class WebContentFetcher:
    """
    Robust web content fetcher with caching and rate limiting.
    
    Provides reliable web content retrieval with built-in caching to speed up
    development and reduce server load, plus configurable rate limiting.
    """
    
    def __init__(self, 
                 cache_directory: Optional[Path] = None,
                 request_timeout: Optional[int] = None,
                 crawl_delay: Optional[float] = None,
                 user_agent: Optional[str] = None):
        """
        Initialize web content fetcher with configuration.
        
        Args:
            cache_directory: Directory for caching downloaded content (uses config default if None)
            request_timeout: Request timeout in seconds (uses config default if None)
            crawl_delay: Delay between requests in seconds (uses config default if None)
            user_agent: User agent string for requests (uses config default if None)
        """
        self.cache_directory = cache_directory or settings.cache_directory
        self.request_timeout = request_timeout or settings.request_timeout_seconds
        self.crawl_delay = crawl_delay or settings.crawl_delay_seconds
        self.user_agent = user_agent or settings.user_agent
        
        # Ensure cache directory exists
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Initialized WebContentFetcher with cache at: {self.cache_directory}")
    
    def _generate_cache_file_path(self, url: str) -> Path:
        """
        Generate cache file path for a given URL.
        
        Args:
            url: URL to generate cache path for
            
        Returns:
            Path to cache file
        """
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:20]
        cache_filename = f"{url_hash}.html"
        return self.cache_directory / cache_filename
    
    def fetch_content(self, url: str, use_cache: bool = True) -> str:
        """
        Fetch web content with caching and rate limiting.
        
        Args:
            url: URL to fetch content from
            use_cache: Whether to use cached content if available
            
        Returns:
            HTML content as string
            
        Raises:
            requests.RequestException: If web request fails
            IOError: If cache operations fail
        """
        cache_file_path = self._generate_cache_file_path(url)
        
        # Try to load from cache first
        if use_cache and cache_file_path.exists():
            try:
                with open(cache_file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    cached_content = file.read()
                logger.debug(f"Loaded content from cache: {url}")
                return cached_content
            except Exception as e:
                logger.warning(f"Failed to read cache for {url}: {e}")
        
        # Fetch from web
        logger.debug(f"Fetching content from web: {url}")
        
        try:
            headers = {"User-Agent": self.user_agent}
            response = requests.get(url, headers=headers, timeout=self.request_timeout)
            response.raise_for_status()
            
            html_content = response.text
            
            # Cache the content
            try:
                with open(cache_file_path, 'w', encoding='utf-8') as file:
                    file.write(html_content)
                logger.debug(f"Cached content for: {url}")
            except Exception as e:
                logger.warning(f"Failed to cache content for {url}: {e}")
            
            # Rate limiting
            if self.crawl_delay > 0:
                time.sleep(self.crawl_delay)
            
            return html_content
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch content from {url}: {e}")
            raise
    
    def clear_cache(self) -> None:
        """Clear all cached content."""
        try:
            for cache_file in self.cache_directory.glob("*.html"):
                cache_file.unlink()
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_statistics(self) -> dict:
        """
        Get statistics about cached content.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            cache_files = list(self.cache_directory.glob("*.html"))
            total_size = sum(file.stat().st_size for file in cache_files)
            
            return {
                "cached_files": len(cache_files),
                "total_size_mb": total_size / (1024 * 1024),
                "cache_directory": str(self.cache_directory)
            }
        except Exception as e:
            logger.error(f"Failed to get cache statistics: {e}")
            return {"cached_files": 0, "total_size_mb": 0.0, "error": str(e)}


class GitLabLinkExtractor:
    """
    Specialized link extractor for GitLab documentation.
    
    Extracts and filters links from HTML content, focusing on GitLab
    handbook and direction pages while excluding non-documentation content.
    """
    
    def __init__(self, 
                 allowed_domains: Optional[Set[str]] = None,
                 excluded_extensions: Optional[tuple] = None):
        """
        Initialize link extractor with filtering configuration.
        
        Args:
            allowed_domains: Set of allowed domains (uses config default if None)
            excluded_extensions: File extensions to exclude (uses config default if None)
        """
        self.allowed_domains = allowed_domains or settings.allowed_domains
        self.excluded_extensions = excluded_extensions or EXCLUDED_FILE_EXTENSIONS
        
        logger.debug(f"Initialized GitLabLinkExtractor for domains: {self.allowed_domains}")
    
    def extract_documentation_links(self, base_url: str, html_content: str) -> Set[str]:
        """
        Extract GitLab documentation links from HTML content.
        
        Args:
            base_url: Base URL for resolving relative links
            html_content: HTML content to extract links from
            
        Returns:
            Set of valid documentation URLs
        """
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            extracted_links = set()
            
            for anchor_tag in soup.find_all("a", href=True):
                href = anchor_tag["href"].strip()
                
                # Skip mailto and fragment links
                if href.startswith(("mailto:", "#")):
                    continue
                
                # Resolve relative URLs
                absolute_url = urljoin(base_url, href)
                parsed_url = urlparse(absolute_url)
                
                # Apply filtering rules
                if self._is_valid_documentation_link(parsed_url):
                    extracted_links.add(absolute_url)
            
            logger.debug(f"Extracted {len(extracted_links)} valid links from {base_url}")
            return extracted_links
            
        except Exception as e:
            logger.error(f"Failed to extract links from {base_url}: {e}")
            return set()
    
    def _is_valid_documentation_link(self, parsed_url) -> bool:
        """
        Check if a parsed URL is a valid GitLab documentation link.
        
        Args:
            parsed_url: Parsed URL object from urlparse
            
        Returns:
            True if URL is valid documentation, False otherwise
        """
        # Check domain
        if parsed_url.netloc not in self.allowed_domains:
            return False
        
        # Check file extensions
        if any(parsed_url.path.lower().endswith(ext) for ext in self.excluded_extensions):
            return False
        
        # Check if it's handbook or direction content
        path = parsed_url.path.lower()
        if not (path.startswith("/handbook/") or path.startswith("/direction/")):
            return False
        
        return True


class HTMLContentExtractor:
    """
    Intelligent HTML content extraction and cleaning.
    
    Extracts meaningful text content from HTML while removing navigation,
    scripts, and other non-content elements.
    """
    
    # HTML tags to completely remove
    NOISE_TAGS = ["script", "style", "noscript", "nav", "footer", "header", "aside"]
    
    def extract_text_content(self, html_content: str) -> str:
        """
        Extract clean text content from HTML.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            Cleaned text content in markdown format
        """
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Remove noise elements
            for tag_name in self.NOISE_TAGS:
                for tag in soup.find_all(tag_name):
                    tag.decompose()
            
            # Convert to markdown for better text structure
            markdown_content = md(str(soup))
            
            # Clean up excessive whitespace
            cleaned_content = re.sub(r"\n{3,}", "\n\n", markdown_content).strip()
            
            logger.debug(f"Extracted {len(cleaned_content)} characters of text content")
            return cleaned_content
            
        except Exception as e:
            logger.error(f"Failed to extract text content: {e}")
            return ""
    
    def get_content_statistics(self, html_content: str) -> dict:
        """
        Get statistics about HTML content.
        
        Args:
            html_content: HTML content to analyze
            
        Returns:
            Dictionary with content statistics
        """
        try:
            text_content = self.extract_text_content(html_content)
            
            return {
                "html_size_bytes": len(html_content),
                "text_size_bytes": len(text_content),
                "compression_ratio": len(text_content) / len(html_content) if html_content else 0,
                "estimated_reading_time_minutes": len(text_content.split()) / 200  # ~200 WPM
            }
        except Exception as e:
            logger.error(f"Failed to get content statistics: {e}")
            return {"html_size_bytes": 0, "text_size_bytes": 0, "error": str(e)}


# Convenience functions for backward compatibility
def fetch(url: str, timeout: Optional[int] = None) -> str:
    """
    Convenience function for fetching web content.
    
    Args:
        url: URL to fetch
        timeout: Request timeout (uses config default if None)
        
    Returns:
        HTML content
    """
    fetcher = WebContentFetcher(request_timeout=timeout)
    return fetcher.fetch_content(url)


def extract_links(base_url: str, html_content: str) -> Set[str]:
    """
    Convenience function for extracting documentation links.
    
    Args:
        base_url: Base URL for resolving relative links
        html_content: HTML content to extract links from
        
    Returns:
        Set of valid documentation URLs
    """
    extractor = GitLabLinkExtractor()
    return extractor.extract_documentation_links(base_url, html_content)


def html_to_text(html_content: str) -> str:
    """
    Convenience function for converting HTML to text.
    
    Args:
        html_content: HTML content to convert
        
    Returns:
        Clean text content
    """
    extractor = HTMLContentExtractor()
    return extractor.extract_text_content(html_content)
