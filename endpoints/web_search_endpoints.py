"""
Web search endpoints for the agentic RAG system.

Copyright (c) 2025 Mike Tallent & Claude Sonnet - Geniusai.biz
All rights reserved. This software is proprietary and confidential.
"""

import logging
from typing import Dict, Any, List
from fastapi import HTTPException
from web.google_search import search_web_content
from ai.openai_integration import generate_openai_embedding
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

async def perform_web_search_for_question(question: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Perform web search for a question and return structured, NUMBERED results for citations.
    Each result gets a stable integer id: 1, 2, 3 ...
    """
    try:
        web_search_enabled = False

        # Check if web search is enabled in context
        if context:
            web_search_enabled = context.get('webSearchEnabled', False)
            if not web_search_enabled and context.get('settings'):
                web_search_enabled = context['settings'].get('webSearchEnabled', False)

        if not web_search_enabled:
            return {
                'web_search_enabled': False,
                'web_results': [],
                'web_content': '',
                'message': 'Web search is disabled'
            }

        logger.info(f"Performing web search for question: {question}")

        # Perform web search (your existing agent)
        search_results = await search_web_content(question, num_results=3)

        if not search_results.get('success'):
            return {
                'web_search_enabled': True,
                'web_results': [],
                'web_content': '',
                'error': search_results.get('error', 'Web search failed'),
                'message': 'Web search failed'
            }

        # Build numbered sources (+ concatenated content for the LLM context)
        web_results: List[Dict[str, Any]] = []
        web_content_parts: List[str] = []
        idx = 1

        for result in search_results.get('scraped_content', []):
            if result.get('success') and result.get('content'):
                item = {
                    'id': idx,
                    'title': result.get('title', '') or 'Untitled',
                    'url': result.get('url', ''),
                    'snippet': result.get('snippet', '') or '',
                    'content_length': len(result.get('content', '')),
                    'relevance_score': float(result.get('relevance_score', 0.5))
                }
                web_results.append(item)

                # Include the number in the stitched context so the model can cite like [1], [2], ...
                content_part = (
                    f"[{item['id']}] {item['title']} ({item['url']})\n"
                    f"{result.get('content','')}\n"
                )
                web_content_parts.append(content_part)
                idx += 1

        combined_web_content = "\n---\n".join(web_content_parts)

        return {
            'web_search_enabled': True,
            'web_results': web_results,
            'web_content': combined_web_content,
            'total_results': len(web_results),
            'search_query': question,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': f'Found {len(web_results)} relevant web sources'
        }

    except Exception as e:
        logger.error(f"Web search error: {e}")
        return {
            'web_search_enabled': True,
            'web_results': [],
            'web_content': '',
            'error': str(e),
            'message': 'Web search encountered an error'
        }

async def enhance_context_with_web_search(question: str, existing_context: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Enhance existing context with web search results.
    """
    try:
        web_search_result = await perform_web_search_for_question(question, context)
        
        if not web_search_result.get('web_search_enabled'):
            return {
                'enhanced_context': existing_context,
                'web_search_used': False,
                'web_sources': []
            }
        
        web_content = web_search_result.get('web_content', '')
        web_results = web_search_result.get('web_results', [])
        
        if web_content:
            # Combine existing context with web content
            enhanced_context = f"{existing_context}\n\n--- WEB SEARCH RESULTS ---\n{web_content}"
            
            return {
                'enhanced_context': enhanced_context,
                'web_search_used': True,
                'web_sources': web_results,
                'web_search_message': web_search_result.get('message', ''),
                'total_web_results': len(web_results)
            }
        else:
            return {
                'enhanced_context': existing_context,
                'web_search_used': True,
                'web_sources': [],
                'web_search_message': web_search_result.get('message', 'No web content found'),
                'total_web_results': 0
            }
            
    except Exception as e:
        logger.error(f"Context enhancement error: {e}")
        return {
            'enhanced_context': existing_context,
            'web_search_used': False,
            'web_sources': [],
            'error': str(e)
        }
from urllib.parse import urlparse

def _shorten(text: str, max_chars: int = 120) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    cut = t[:max_chars].rstrip()
    i = cut.rfind(" ")
    return (cut[:i] if i > max_chars * 0.8 else cut) + "…"

def _first_sentence(s: str, max_chars: int = 220) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    for sep in [". ", "。", "! ", "? "]:
        if sep in s:
            s = s.split(sep, 1)[0]
            break
    return _shorten(s, max_chars).rstrip(".!?") + "."

def _two_sentence_explainer(src: dict, question: str) -> str:
    title = (src.get("title") or "").strip()
    url = src.get("url") or ""
    domain = urlparse(url).netloc.replace("www.", "") if url else ""
    snippet = (src.get("snippet") or "").strip()

    s1 = _first_sentence(snippet, max_chars=220) if snippet else (
        f"{title or domain or 'This source'} provides background and concrete details."
    )
    s2 = f"It’s relevant because it directly relates to: {_shorten(question, 120)}"
    return f"{s1} {s2}"

def _title_with_expl(src: dict, question: str) -> str:
    base = (src.get("title") or "").strip()
    expl = _two_sentence_explainer(src, question)
    return (base if base else "Untitled") + "\n" + expl
import re

def is_new_question(user_text: str, last_entities: list[str], llm_call) -> bool:
    """
    Returns:
      True  -> NEW question (run web search)
      False -> FOLLOW-UP (reuse previous sources)

    llm_call(prompt: str) -> str
      Should return either "NEW_INFO_REQUEST" or "FOLLOW_UP_EXPAND"
    """
    t = user_text.strip().lower()

    # --- Heuristic patterns ---
    follow_up_patterns = [
        r"\b(explain|expand|elaborate|clarify|summarize|simplify|details?)\b",
        r"\b(go deeper|step[- ]by[- ]step|break it down)\b",
        r"\b(examples?)\b",
        r"\b(what do you mean|what does that mean)\b",
    ]
    deictics = [r"\b(this|that|above|previous|earlier|it)\b"]
    new_info_hints = [
        r"\b(latest|today|now|currently|as of|update|recent|new)\b",
        r"\b(compare|versus|vs\.?)\b",
        r"\b(price|availability|stock|release date|schedule|score|result)\b",
        r"\b(after|since)\s+(20\d{2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",
    ]
    entity_re = re.compile(r"\b([A-Z][A-Za-z0-9\-]{2,}(?:\s+[A-Z][A-Za-z0-9\-]{2,})*)\b")

    def any_match(patterns, text): 
        return any(re.search(p, text, flags=re.I) for p in patterns)

    def extract_entities(text: str) -> list[str]:
        ents = [m.group(1).strip() for m in entity_re.finditer(text)]
        seen, out = set(), []
        for e in ents:
            if e not in seen and len(e.split()) <= 6:
                seen.add(e); out.append(e)
        return out

    # --- if / elif / else routing ---
    if (any_match(follow_up_patterns, t) and any_match(deictics, t)) or (
        len(t.split()) <= 6 and any_match(deictics, t)
    ):
        # FOLLOW-UP → False
        return False
    elif any_match(new_info_hints, t) or any(e not in set(last_entities) for e in extract_entities(user_text)):
        # NEW info → True
        return True
    else:
        # LLM gate (must return "NEW_INFO_REQUEST" or "FOLLOW_UP_EXPAND")
        prompt = f"""Decide if the user asks to EXPAND the last answer (FOLLOW_UP_EXPAND)
        or requests NEW external info (NEW_INFO_REQUEST). Respond with exactly one token only return True if the question is NEW_INFO_REQUEST or False if the question is FOLLOW_UP_EXPAND
        .

        Previous entities: {', '.join(last_entities) or 'None'}
        User: {user_text}"""
        label = (llm_call(prompt) or "").strip()
        return label == "NEW_INFO_REQUEST"
