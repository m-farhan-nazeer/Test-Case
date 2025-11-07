"""
Google Search Agent for web scraping and search functionality.

Copyright (c) 2025 Mike Tallent & Claude Sonnet - Geniusai.biz
All rights reserved. This software is proprietary and confidential.
"""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Any
from urllib.parse import quote_plus, urlparse
from bs4 import BeautifulSoup
import re
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class GoogleSearchAgent:
    """Advanced Google Search Agent with web scraping capabilities."""
    
    def __init__(self):
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        # Domains that are often low-signal for generic searches (directories, social profiles, pinboards)
        self._default_blocklist = {
            "linkedin.com", "facebook.com", "m.facebook.com", "twitter.com", "x.com",
            "instagram.com", "pinterest.com", "tiktok.com", "t.co", "linktr.ee"
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    # ---------------------------
    # SEARCH
    # ---------------------------
    async def search_google(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search using DuckDuckGo and Bing as more reliable alternatives to Google scraping.
        Adds a sensible domain blocklist to avoid directory/social results unless explicitly requested.
        """
        try:
            if not self.session:
                raise RuntimeError("GoogleSearchAgent must be used as async context manager")
            
            logger.info(f"Searching web for: {query}")
            encoded_query = quote_plus(query)

            search_urls = [
                f"https://duckduckgo.com/html/?q={encoded_query}",
                f"https://www.bing.com/search?q={encoded_query}",
            ]
            
            results: List[Dict[str, Any]] = []
            want_social = any(token in query.lower() for token in ["linkedin", "twitter", "x.com", "facebook", "instagram", "pinterest", "tiktok"])
            
            for search_url in search_urls:
                try:
                    logger.info(f"Trying search URL: {search_url}")
                    
                    headers = {
                        **self.headers,
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Cache-Control': 'no-cache',
                        'Pragma': 'no-cache',
                        'Sec-Fetch-Dest': 'document',
                        'Sec-Fetch-Mode': 'navigate',
                        'Sec-Fetch-Site': 'none',
                        'Sec-Fetch-User': '?1',
                        'Referer': 'https://duckduckgo.com/'
                    }
                    
                    async with self.session.get(search_url, headers=headers) as response:
                        if response.status != 200:
                            logger.warning(f"Search failed with status: {response.status}")
                            continue
                        
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # ---- DuckDuckGo parsing
                        if 'duckduckgo.com' in search_url:
                            search_results = soup.find_all('div', class_='result')
                            for result in search_results:
                                try:
                                    title_elem = result.find('a', class_='result__a')
                                    if not title_elem:
                                        continue
                                    title = (title_elem.get_text() or "").strip()
                                    url = title_elem.get('href')
                                    
                                    # DDG sometimes uses protocol-relative URLs or redirects; normalize
                                    if not url:
                                        continue
                                    if url.startswith('//'):
                                        url = 'https:' + url
                                    if not url.startswith('http'):
                                        continue
                                    
                                    snippet_elem = result.find('a', class_='result__snippet')
                                    snippet = (snippet_elem.get_text().strip() if snippet_elem else "")
                                    
                                    domain = urlparse(url).netloc.lower()
                                    if not want_social and any(bad in domain for bad in self._default_blocklist):
                                        continue
                                    
                                    results.append({
                                        'title': title or domain,
                                        'url': url,
                                        'snippet': snippet,
                                        'search_query': query,
                                        'timestamp': datetime.now(timezone.utc).isoformat(),
                                        'source': 'duckduckgo'
                                    })
                                    
                                    if len(results) >= num_results:
                                        break
                                except Exception as e:
                                    logger.warning(f"Error parsing DuckDuckGo result: {e}")
                                    continue
                        
                        # ---- Bing parsing
                        elif 'bing.com' in search_url:
                            search_results = soup.find_all('li', class_='b_algo')
                            for result in search_results:
                                try:
                                    h2 = result.find('h2')
                                    link_elem = h2.find('a') if h2 else None
                                    if not link_elem:
                                        continue
                                    title = (link_elem.get_text() or "").strip()
                                    url = link_elem.get('href')
                                    if not url or not url.startswith('http'):
                                        continue
                                    
                                    # Snippet may be in <p> or .b_caption
                                    snippet_elem = result.find('p') or result.find('div', class_='b_caption')
                                    snippet = (snippet_elem.get_text().strip() if snippet_elem else "")
                                    
                                    domain = urlparse(url).netloc.lower()
                                    if not want_social and any(bad in domain for bad in self._default_blocklist):
                                        continue
                                    
                                    results.append({
                                        'title': title or domain,
                                        'url': url,
                                        'snippet': snippet,
                                        'search_query': query,
                                        'timestamp': datetime.now(timezone.utc).isoformat(),
                                        'source': 'bing'
                                    })
                                    
                                    if len(results) >= num_results:
                                        break
                                except Exception as e:
                                    logger.warning(f"Error parsing Bing result: {e}")
                                    continue
                        
                        if len(results) >= num_results:
                            break
                            
                except Exception as e:
                    logger.warning(f"Error with search URL {search_url}: {e}")
                    continue
            
            if not results:
                logger.info("No search results found, generating fallback URLs")
                results = self._generate_fallback_urls(query)
            
            logger.info(f"Found {len(results)} search results for query: {query}")
            return results[:num_results]
                
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return self._generate_fallback_urls(query)
    
    def _generate_fallback_urls(self, query: str) -> List[Dict[str, Any]]:
        """
        Generate neutral, query-driven fallback URLs when search engines fail.
        (Removed agriculture-specific paths.)
        """
        fallback_sites = [
            "bbc.com",
            "cnn.com", 
            "reuters.com",
            "wikipedia.org",
            "news.google.com"
        ]
        
        results: List[Dict[str, Any]] = []
        encoded_query = quote_plus(query)
        
        for site in fallback_sites:
            url = f"https://www.{site}/search?q={encoded_query}" if not site.startswith("news.") else f"https://{site}/search?q={encoded_query}"
            title = f"Search results for '{query}' - {site.upper()}"
            results.append({
                'title': title,
                'url': url,
                'snippet': f"Search results related to: {query}",
                'search_query': query,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'source': 'fallback'
            })
        
        return results[:3]
    
    # ---------------------------
    # SCRAPE
    # ---------------------------
    async def scrape_url_content(self, url: str, max_length: int = 5000) -> Dict[str, Any]:
        """
        Scrape content from a URL and return structured data.
        Uses neutral fallback text (no agriculture), derived from domain if needed.
        """
        try:
            if not self.session:
                raise RuntimeError("GoogleSearchAgent must be used as async context manager")
            
            logger.info(f"Scraping content from: {url}")
            
            scraping_headers = {
                **self.headers,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache',
                'Referer': 'https://duckduckgo.com/',
            }
            
            async with self.session.get(url, headers=scraping_headers, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status != 200:
                    logger.warning(f"Failed to scrape {url}, status: {response.status}")
                    fallback_content = self._generate_fallback_content(url)
                    return {
                        'url': url,
                        'title': fallback_content['title'],
                        'content': fallback_content['content'],
                        'error': f'HTTP {response.status}',
                        'success': True,
                        'fallback': True
                    }
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove unwanted elements
                for element in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
                    element.decompose()
                
                # Title
                title = soup.find('title')
                title_text = title.get_text().strip() if title else urlparse(url).netloc
                
                # Content extraction strategies
                content = ""
                content_selectors = [
                    'article', 'main', '[role="main"]',
                    '.content', '.post-content', '.entry-content', 
                    '.article-content', '.story-body', '.article-body',
                    '.post-body', '.content-body'
                ]
                
                for selector in content_selectors:
                    content_elem = soup.select_one(selector)
                    if content_elem:
                        content = content_elem.get_text(separator=' ', strip=True)
                        if len(content) > 200:
                            break
                
                if not content or len(content) < 200:
                    paragraphs = soup.find_all('p')
                    if paragraphs:
                        content = ' '.join([p.get_text(strip=True) for p in paragraphs[:12]])
                
                if not content or len(content) < 100:
                    body = soup.find('body')
                    if body:
                        content = body.get_text(separator=' ', strip=True)
                
                content = re.sub(r'\s+', ' ', content or '').strip()
                
                if not content or len(content) < 50:
                    fallback_content = self._generate_fallback_content(url)
                    return {
                        'url': url,
                        'title': fallback_content['title'],
                        'content': fallback_content['content'],
                        'length': len(fallback_content['content']),
                        'success': True,
                        'fallback': True,
                        'scraped_at': datetime.now(timezone.utc).isoformat()
                    }
                
                if len(content) > max_length:
                    content = content[:max_length] + "..."
                
                return {
                    'url': url,
                    'title': title_text,
                    'content': content,
                    'length': len(content),
                    'success': True,
                    'scraped_at': datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            fallback_content = self._generate_fallback_content(url)
            return {
                'url': url,
                'title': fallback_content['title'],
                'content': fallback_content['content'],
                'length': len(fallback_content['content']),
                'success': True,
                'fallback': True,
                'error': str(e),
                'scraped_at': datetime.now(timezone.utc).isoformat()
            }
    
    def _generate_fallback_content(self, url: str) -> Dict[str, str]:
        """
        Neutral fallback content when scraping fails (no hard-coded agriculture text).
        """
        domain = urlparse(url).netloc or url
        domain = domain.lower()
        return {
            'title': f'{domain}',
            'content': (
                "We were unable to extract article text from this page. "
                "This placeholder is shown when the site blocks scraping or returns minimal content. "
                "Try opening the URL directly for details."
            )
        }
    
    # ---------------------------
    # ORCHESTRATION
    # ---------------------------
    async def search_and_scrape(self, query: str, num_results: int = 3, max_content_length: int = 3000) -> Dict[str, Any]:
        """
        Perform web search and scrape content from the top results.
        """
        try:
            search_results = await self.search_google(query, num_results)
            
            if not search_results:
                return {
                    'query': query,
                    'search_results': [],
                    'scraped_content': [],
                    'total_results': 0,
                    'success': False,
                    'error': 'No search results found'
                }
            
            scraping_tasks = [
                self.scrape_url_content(result['url'], max_content_length)
                for result in search_results
            ]
            
            scraped_content: List[Dict[str, Any]] = []
            scraped_results = await asyncio.gather(*scraping_tasks, return_exceptions=True)
            
            for i, scraped in enumerate(scraped_results):
                if isinstance(scraped, Exception):
                    logger.error(f"Scraping task failed: {scraped}")
                    continue
                
                if scraped.get('success') and scraped.get('content'):
                    combined_result = {
                        **search_results[i],
                        **scraped,
                        'relevance_score': 1.0 - (i * 0.1)
                    }
                    scraped_content.append(combined_result)
            
            return {
                'query': query,
                'search_results': search_results,
                'scraped_content': scraped_content,
                'total_results': len(scraped_content),
                'success': True,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Search and scrape error: {e}")
            return {
                'query': query,
                'search_results': [],
                'scraped_content': [],
                'total_results': 0,
                'success': False,
                'error': str(e)
            }

# Convenience function for easy usage
async def search_web_content(query: str, num_results: int = 3) -> Dict[str, Any]:
    """
    Convenience function to search and scrape web content.
    """
    async with GoogleSearchAgent() as agent:
        return await agent.search_and_scrape(query, num_results)
