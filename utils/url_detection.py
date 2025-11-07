"""
URL detection and extraction utilities.

Copyright (c) 2025 Mike Tallent & Claude Sonnet - Geniusai.biz
All rights reserved. This software is proprietary and confidential.
"""

import re
import logging
from typing import List, Dict, Any
from urllib.parse import urlparse, urljoin

logger = logging.getLogger(__name__)


def extract_urls_from_text(text: str) -> List[str]:
    """Extract URLs from text using regex patterns."""
    # Comprehensive URL regex pattern
    url_pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'
    
    # Also match URLs without protocol
    url_pattern_no_protocol = r'(?:www\.)?[-\w.]+\.(?:com|org|net|edu|gov|mil|int|co|uk|de|fr|jp|au|ca|us|info|biz|name|mobi|travel|museum|aero|coop|jobs|tel|xxx|asia|cat|pro|post|geo|xxx|arpa|[a-z]{2})(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'
    
    urls = []
    
    # Find URLs with protocol
    protocol_urls = re.findall(url_pattern, text, re.IGNORECASE)
    urls.extend(protocol_urls)
    
    # Find URLs without protocol and add https://
    no_protocol_urls = re.findall(url_pattern_no_protocol, text, re.IGNORECASE)
    for url in no_protocol_urls:
        if not url.startswith(('http://', 'https://')):
            if url.startswith('www.'):
                urls.append(f'https://{url}')
            else:
                # Check if it looks like a domain
                if '.' in url and not url.startswith('http'):
                    urls.append(f'https://{url}')
    
    # Remove duplicates while preserving order
    unique_urls = []
    seen = set()
    for url in urls:
        if url not in seen:
            unique_urls.append(url)
            seen.add(url)
    
    return unique_urls


def validate_url(url: str) -> bool:
    """Validate if a URL is properly formatted and accessible."""
    try:
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)
    except Exception:
        return False


def normalize_url(url: str) -> str:
    """Normalize URL by ensuring it has a proper protocol."""
    if not url.startswith(('http://', 'https://')):
        if url.startswith('www.'):
            return f'https://{url}'
        elif '.' in url:
            return f'https://{url}'
    return url


def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return ""


def is_same_domain(url1: str, url2: str) -> bool:
    """Check if two URLs are from the same domain."""
    try:
        domain1 = extract_domain(url1)
        domain2 = extract_domain(url2)
        return domain1.lower() == domain2.lower()
    except Exception:
        return False


def analyze_query_for_urls(query: str) -> Dict[str, Any]:
    """Analyze a query for URLs and return detailed information."""
    urls = extract_urls_from_text(query)
    
    analysis = {
        "has_urls": len(urls) > 0,
        "url_count": len(urls),
        "urls": [],
        "domains": [],
        "query_without_urls": query
    }
    
    if urls:
        # Remove URLs from query text
        query_without_urls = query
        for url in urls:
            query_without_urls = query_without_urls.replace(url, "").strip()
        
        # Clean up extra spaces
        query_without_urls = re.sub(r'\s+', ' ', query_without_urls).strip()
        analysis["query_without_urls"] = query_without_urls
        
        # Process each URL
        for url in urls:
            normalized_url = normalize_url(url)
            if validate_url(normalized_url):
                domain = extract_domain(normalized_url)
                analysis["urls"].append({
                    "original": url,
                    "normalized": normalized_url,
                    "domain": domain,
                    "valid": True
                })
                if domain not in analysis["domains"]:
                    analysis["domains"].append(domain)
            else:
                analysis["urls"].append({
                    "original": url,
                    "normalized": normalized_url,
                    "domain": "",
                    "valid": False
                })
    
    return analysis
