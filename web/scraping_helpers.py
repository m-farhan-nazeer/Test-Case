"""
Web scraping helper functions for the agentic RAG system.

Copyright (c) 2025 Mike Tallent & Claude Sonnet - Geniusai.biz
All rights reserved. This software is proprietary and confidential.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from datetime import datetime, timezone
from typing import Dict, Any, List, Set
import re
import logging

logger = logging.getLogger(__name__)


def get_scraping_headers() -> Dict[str, str]:
    """Get standard headers for web scraping requests."""
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }


def clean_text_content(content_text: str) -> str:
    """Clean and format extracted text content."""
    lines = content_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line and len(line) > 3:  # Skip very short lines
            # Remove excessive whitespace
            line = re.sub(r'\s+', ' ', line)
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def extract_title_from_soup(soup: BeautifulSoup, url: str) -> str:
    """Extract title from BeautifulSoup object with fallbacks."""
    title = ""
    try:
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find('h1'):
            h1_text = soup.find('h1').get_text()
            if h1_text:
                title = h1_text.strip()
        
        # Fallback if no title found or title is empty
        if not title:
            parsed_url = urlparse(url)
            title = f"Content from {parsed_url.netloc}"
            
    except Exception as e:
        logger.warning(f"Error extracting title from {url}: {e}")
        parsed_url = urlparse(url)
        title = f"Content from {parsed_url.netloc}"
    
    return title


def extract_main_content(soup: BeautifulSoup) -> str:
    """Extract main content from BeautifulSoup object."""
    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
        script.decompose()
    
    # Extract main content
    content_selectors = [
        'main', 'article', '.content', '#content', '.main-content', 
        '.post-content', '.entry-content', '.article-content',
        '.page-content', '.blog-content', '.story-content'
    ]
    
    main_content = None
    for selector in content_selectors:
        main_content = soup.select_one(selector)
        if main_content:
            break
    
    # If no main content found, use body
    if not main_content:
        main_content = soup.find('body')
    
    if not main_content:
        main_content = soup
    
    # Extract text content
    content_text = main_content.get_text(separator='\n', strip=True)
    
    return clean_text_content(content_text)


def extract_summary_from_soup(soup: BeautifulSoup, content: str) -> str:
    """Extract summary/description from BeautifulSoup object."""
    summary = ""
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc:
        summary = meta_desc.get('content', '').strip()
    elif soup.find('meta', attrs={'property': 'og:description'}):
        summary = soup.find('meta', attrs={'property': 'og:description'}).get('content', '').strip()
    
    # If no summary found, use first paragraph or first 200 chars
    if not summary and content:
        first_paragraph = content.split('\n')[0] if content else ""
        summary = first_paragraph[:200] + "..." if len(first_paragraph) > 200 else first_paragraph
    
    return summary


def extract_metadata_from_soup(soup: BeautifulSoup, url: str, response_status: int, content_length: int) -> Dict[str, Any]:
    """Extract metadata from BeautifulSoup object."""
    metadata = {
        "url": url,
        "domain": urlparse(url).netloc,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "content_type": "web_scraping",
        "response_status": response_status,
        "content_length": content_length
    }
    
    # Extract Open Graph data if available
    og_title = soup.find('meta', attrs={'property': 'og:title'})
    if og_title:
        metadata['og_title'] = og_title.get('content', '')
    
    og_type = soup.find('meta', attrs={'property': 'og:type'})
    if og_type:
        metadata['og_type'] = og_type.get('content', '')
    
    # Extract keywords if available
    keywords_meta = soup.find('meta', attrs={'name': 'keywords'})
    if keywords_meta:
        metadata['keywords'] = keywords_meta.get('content', '')
    
    return metadata


def get_skip_extensions() -> Set[str]:
    """Get file extensions to skip during scraping."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.avif', '.svg', '.bmp', '.tiff', '.ico'}
    media_extensions = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mp3', '.wav', '.ogg', '.pdf', '.zip', '.rar', '.exe', '.dmg'}
    return image_extensions.union(media_extensions)


def should_skip_url(url: str) -> bool:
    """Check if URL should be skipped based on file extension."""
    parsed_url = urlparse(url)
    path_lower = parsed_url.path.lower()
    skip_extensions = get_skip_extensions()
    return any(path_lower.endswith(ext) for ext in skip_extensions)


def should_skip_content_type(content_type: str) -> bool:
    """Check if content type should be skipped."""
    content_type_lower = content_type.lower()
    skip_types = ['image/', 'video/', 'audio/', 'application/pdf', 'application/zip']
    return any(skip_type in content_type_lower for skip_type in skip_types)


def discover_internal_links(soup: BeautifulSoup, base_url: str, base_domain: str) -> List[str]:
    """Discover internal links from a webpage."""
    internal_links = set()
    skip_extensions = get_skip_extensions()
    
    # Find all links
    for link in soup.find_all('a', href=True):
        href = link['href'].strip()
        
        # Skip empty links, anchors, and javascript
        if not href or href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
            continue
        
        # Convert relative URLs to absolute
        absolute_url = urljoin(base_url, href)
        parsed_url = urlparse(absolute_url)
        
        # Skip URLs that end with image or media file extensions
        path_lower = parsed_url.path.lower()
        if any(path_lower.endswith(ext) for ext in skip_extensions):
            continue
        
        # Only include links from the same domain
        if parsed_url.netloc == base_domain:
            # Remove fragments and query parameters for deduplication
            clean_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
            if clean_url != base_url and clean_url not in internal_links:
                internal_links.add(clean_url)
    
    return list(internal_links)


async def scrape_website_content(url: str) -> Dict[str, Any]:
    """Scrape content from a website URL using BeautifulSoup."""
    try:
        headers = get_scraping_headers()
        
        # Make the request with timeout
        response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        response.raise_for_status()
        
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract components
        title = extract_title_from_soup(soup, url)
        content = extract_main_content(soup)
        summary = extract_summary_from_soup(soup, content)
        metadata = extract_metadata_from_soup(soup, url, response.status_code, len(content))
        
        # Validate content
        if not content or len(content.strip()) < 50:
            raise ValueError("Insufficient content extracted from the webpage")
        
        return {
            "title": title,
            "content": content,
            "summary": summary,
            "metadata": metadata
        }
        
    except requests.exceptions.Timeout:
        logger.error(f"Timeout while scraping URL: {url}")
        raise Exception("Request timeout while accessing the website")
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error while scraping URL: {url}")
        raise Exception("Unable to connect to the website")
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error while scraping URL: {url}, Status: {e.response.status_code}")
        raise Exception(f"Website returned error: {e.response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error while scraping URL: {url}, Error: {str(e)}")
        raise Exception(f"Error accessing website: {str(e)}")
    except ValueError as e:
        logger.error(f"Content extraction error for URL: {url}, Error: {str(e)}")
        raise Exception(str(e))
    except Exception as e:
        logger.error(f"Unexpected error while scraping URL: {url}, Error: {str(e)}")
        raise Exception(f"Unexpected error during web scraping: {str(e)}")


async def bulk_scrape_website(base_url: str, max_pages: int = 10, title_prefix: str = None, 
                            summary_prefix: str = None, combine_into_single_document: bool = True) -> Dict[str, Any]:
    """Scrape multiple pages from a website by discovering internal links."""
    try:
        parsed_base = urlparse(base_url)
        base_domain = parsed_base.netloc
        
        scraped_urls = []
        failed_urls = []
        discovered_urls = {base_url}
        processed_urls = set()
        
        # For combined document mode
        combined_content_parts = []
        combined_metadata = {
            "url": base_url,
            "domain": base_domain,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "content_type": "bulk_web_scraping_combined",
            "total_pages_planned": max_pages,
            "scraped_pages": []
        }
        
        headers = get_scraping_headers()
        
        # Process URLs until we reach max_pages or run out of URLs
        while len(scraped_urls) < max_pages and discovered_urls:
            # Get next URL to process
            current_url = discovered_urls.pop()
            
            if current_url in processed_urls:
                continue
                
            processed_urls.add(current_url)
            
            try:
                logger.info(f"Scraping page {len(scraped_urls) + 1}/{max_pages}: {current_url}")
                
                # Skip URLs that look like images or media files
                if should_skip_url(current_url):
                    failed_urls.append({
                        "url": current_url,
                        "error": "Skipped: Image or media file"
                    })
                    continue
                
                # Make request
                response = requests.get(current_url, headers=headers, timeout=30, allow_redirects=True)
                response.raise_for_status()
                
                # Check content type to avoid processing images that might not have file extensions
                content_type = response.headers.get('content-type', '').lower()
                if should_skip_content_type(content_type):
                    failed_urls.append({
                        "url": current_url,
                        "error": "Skipped: Non-text content type"
                    })
                    continue
                
                # Parse content
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Discover more internal links from this page
                if len(scraped_urls) < max_pages - 1:  # Only discover more if we haven't reached the limit
                    new_links = discover_internal_links(soup, current_url, base_domain)
                    for link in new_links:
                        if link not in processed_urls and len(discovered_urls) + len(scraped_urls) < max_pages * 2:
                            discovered_urls.add(link)
                
                # Extract content using helper functions
                title = extract_title_from_soup(soup, current_url)
                content = extract_main_content(soup)
                
                # Add prefix if provided
                if title_prefix:
                    title = f"{title_prefix} - {title}"
                
                # Skip pages with insufficient content
                if not content or len(content.strip()) < 50:
                    failed_urls.append({
                        "url": current_url,
                        "error": "Insufficient content extracted"
                    })
                    continue
                
                # Extract summary
                summary = extract_summary_from_soup(soup, content)
                if summary_prefix:
                    summary = f"{summary_prefix} {summary}"
                
                if combine_into_single_document:
                    # Add content to combined document
                    page_header = f"\n\n=== PAGE {len(scraped_urls) + 1}: {title} ===\nURL: {current_url}\n\n"
                    combined_content_parts.append(page_header + content)
                    
                    # Track page metadata
                    combined_metadata["scraped_pages"].append({
                        "url": current_url,
                        "title": title,
                        "content_length": len(content),
                        "page_number": len(scraped_urls) + 1
                    })
                    
                    scraped_urls.append(current_url)
                    logger.info(f"Successfully scraped page {len(scraped_urls)}: {title}")
                else:
                    # Return individual page data for separate document creation
                    scraped_urls.append({
                        "url": current_url,
                        "title": title,
                        "content": content,
                        "summary": summary,
                        "metadata": extract_metadata_from_soup(soup, current_url, response.status_code, len(content))
                    })
                
            except Exception as e:
                logger.error(f"Error scraping {current_url}: {str(e)}")
                failed_urls.append({
                    "url": current_url,
                    "error": str(e)
                })
        
        result = {
            "total_pages": len(scraped_urls) + len(failed_urls),
            "successful_pages": len(scraped_urls),
            "failed_pages": len(failed_urls),
            "scraped_urls": scraped_urls if not combine_into_single_document else [url for url in scraped_urls if isinstance(url, str)],
            "failed_urls": failed_urls,
            "base_url": base_url,
            "base_domain": base_domain,
            "combined_into_single_document": combine_into_single_document
        }
        
        if combine_into_single_document and combined_content_parts:
            result["combined_content"] = "\n".join(combined_content_parts)
            result["combined_metadata"] = combined_metadata
            result["combined_title"] = title_prefix or f"Complete Website Content - {base_domain}"
            result["combined_summary"] = summary_prefix or f"Combined content from {len(scraped_urls)} pages scraped from {base_domain}"
        
        return result
        
    except Exception as e:
        logger.error(f"Error in bulk scraping: {str(e)}")
        raise Exception(f"Bulk scraping error: {str(e)}")
