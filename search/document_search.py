"""
Document search functions for in-memory and database operations.

Copyright (c) 2025 Mike Tallent & Claude Sonnet - Geniusai.biz
All rights reserved. This software is proprietary and confidential.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def search_documents_in_memory(question: str, search_terms: List[str], in_memory_documents: List[Dict], limit: int = 5) -> List[Dict]:
    """Search documents in memory with consistent logic."""
    scored_docs = []
    for doc in in_memory_documents:
        content_lower = doc["content"].lower()
        title_lower = doc["title"].lower()
        summary_lower = (doc["summary"] or "").lower()
        metadata_lower = str(doc.get("metadata", {})).lower()
        
        score = 0
        for term in search_terms:
            if term in title_lower:
                score += 3
            if term in summary_lower:
                score += 2
            if term in content_lower:
                score += 1
            if term in metadata_lower:
                score += 1
        
        if score > 0:
            scored_docs.append((doc, score))
    
    if scored_docs:
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:limit]]
    else:
        return in_memory_documents[-limit:] if in_memory_documents else []
