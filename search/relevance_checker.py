"""
Document relevance checking utilities.

Copyright (c) 2025 Mike Tallent & Claude Sonnet - Geniusai.biz
All rights reserved. This software is proprietary and confidential.
"""

import json
import logging
import hashlib
from typing import List, Dict, Any
from ai.openai_integration import generate_openai_embedding
from content.extraction_functions import chunk_document_content, calculate_chunk_similarities
from vector.similarity_functions import calculate_cosine_similarity
from utils.utility_functions import extract_important_terms

logger = logging.getLogger(__name__)


async def check_question_document_relevance(question: str, documents: List[Dict], relevance_threshold: float = 0.2) -> Dict[str, Any]:
    """Enhanced question-document relevance checking with chunk-based similarity scoring."""
    if not documents:
        return {"is_relevant": False, "max_relevance": 0.0, "relevant_docs": []}
    
    # Generate embedding for the question with caching
    question_cache_key = f"question:{hashlib.md5(question.encode()).hexdigest()}"
    question_embedding = await generate_openai_embedding(question, question_cache_key)
    question_words = set(question.lower().split())
    question_keywords = set(word for word in question_words if len(word) > 3)
    
    relevant_docs = []
    max_relevance = 0.0
    
    for doc in documents:
        # Get document content for chunk analysis
        try:
            doc_content = doc.get("content", "") if hasattr(doc, 'get') else (doc["content"] if "content" in doc else "")
            doc_title = doc.get("title", "") if hasattr(doc, 'get') else (doc["title"] if "title" in doc else "")
            doc_summary = doc.get("summary", "") if hasattr(doc, 'get') else (doc["summary"] if "summary" in doc and doc["summary"] else "")
            doc_id = doc.get("id", "") if hasattr(doc, 'get') else (doc["id"] if "id" in doc else "")
        except (KeyError, TypeError):
            doc_content = ""
            doc_title = ""
            doc_summary = ""
            doc_id = ""
        
        # Get or generate document embedding with caching
        try:
            doc_embedding_val = doc.get("embedding", "") if hasattr(doc, 'get') else (doc["embedding"] if "embedding" in doc else "")
        except (KeyError, TypeError):
            doc_embedding_val = ""
            
        if doc_embedding_val:
            # Handle JSON string embeddings from SQLite
            if isinstance(doc_embedding_val, str):
                try:
                    doc_embedding = json.loads(doc_embedding_val)
                    if not isinstance(doc_embedding, list) or len(doc_embedding) == 0:
                        doc_embedding = None
                except (json.JSONDecodeError, TypeError):
                    doc_embedding = None
            else:
                doc_embedding = doc_embedding_val if isinstance(doc_embedding_val, list) else None
        else:
            doc_embedding = None
            
        if doc_embedding is None:
            # Generate embedding for document content with caching
            doc_text = f"{doc_title} {doc_content} {doc_summary}"
            doc_cache_key = f"doc:{doc_id}:{hashlib.md5(doc_text.encode()).hexdigest()}"
            doc_embedding = await generate_openai_embedding(doc_text, doc_cache_key)
        
        # Calculate overall document similarity
        overall_semantic_similarity = calculate_cosine_similarity(question_embedding, doc_embedding)
        
        # Chunk-based analysis for better relevance detection
        max_chunk_similarity = 0.0
        best_chunk_score = 0.0
        
        if len(doc_content) > 300:  # Only chunk longer documents
            chunks = chunk_document_content(doc_content, chunk_size=400, overlap=50)
            if chunks:
                chunk_similarities = await calculate_chunk_similarities(question, chunks, question_embedding)
                if chunk_similarities:
                    max_chunk_similarity = chunk_similarities[0]["semantic_similarity"]
                    best_chunk_score = chunk_similarities[0]["combined_score"]
        
        # Use the better of overall or chunk-based similarity
        semantic_similarity = max(overall_semantic_similarity, max_chunk_similarity)
        
        # Enhanced keyword matching with different weights for title vs content
        title_words = set(doc_title.lower().split())
        content_words = set(doc_content.lower().split())
        summary_words = set(doc_summary.lower().split())
        
        title_overlap = len(question_keywords.intersection(title_words)) / max(len(question_keywords), 1)
        content_overlap = len(question_keywords.intersection(content_words)) / max(len(question_keywords), 1)
        summary_overlap = len(question_keywords.intersection(summary_words)) / max(len(question_keywords), 1)
        
        # Weighted keyword score (title matches are more important)
        keyword_score = (title_overlap * 0.5) + (content_overlap * 0.3) + (summary_overlap * 0.2)
        
        # Calculate phrase matching for better context understanding
        question_phrases = []
        question_tokens = question.lower().split()
        for i in range(len(question_tokens) - 1):
            question_phrases.append(f"{question_tokens[i]} {question_tokens[i+1]}")
        
        doc_text_lower = f"{doc_title} {doc_content} {doc_summary}".lower()
        phrase_matches = sum(1 for phrase in question_phrases if phrase in doc_text_lower)
        phrase_score = phrase_matches / max(len(question_phrases), 1)
        
        # Content quality factor
        content_length_factor = min(1.0, len(doc_content) / 1000)
        
        # Question type analysis for better matching
        question_lower = question.lower()
        is_specific_query = any(word in question_lower for word in ['what', 'how', 'when', 'where', 'who', 'why'])
        specificity_bonus = 0.1 if is_specific_query and (title_overlap > 0.3 or phrase_score > 0.2) else 0
        
        # Combined relevance score with enhanced weighting
        combined_relevance = (
            semantic_similarity * 0.45 +      # Semantic understanding (slightly reduced)
            keyword_score * 0.25 +            # Keyword matching
            phrase_score * 0.15 +             # Phrase matching
            best_chunk_score * 0.1 +          # Best chunk relevance
            content_length_factor * 0.05 +    # Content comprehensiveness
            specificity_bonus                 # Bonus for specific queries
        )
        
        # Dynamic threshold based on query complexity and document quality
        base_threshold = max(relevance_threshold, 0.25)
        dynamic_threshold = base_threshold
        
        # Lower threshold for high-quality semantic matches
        if semantic_similarity >= 0.7:
            dynamic_threshold = max(0.2, base_threshold - 0.1)
        # Raise threshold for keyword-only matches
        elif semantic_similarity < 0.4 and keyword_score < 0.3:
            dynamic_threshold = base_threshold + 0.1
        
        # Additional checks to avoid false positives
        passes_quality_check = (
            (combined_relevance >= dynamic_threshold) and
            (semantic_similarity >= 0.25 or keyword_score >= 0.3 or phrase_score >= 0.2)
        )
        
        if passes_quality_check:
            relevant_docs.append({
                "doc": doc,
                "relevance": combined_relevance,
                "semantic_similarity": semantic_similarity,
                "overall_semantic_similarity": overall_semantic_similarity,
                "max_chunk_similarity": max_chunk_similarity,
                "best_chunk_score": best_chunk_score,
                "keyword_score": keyword_score,
                "phrase_score": phrase_score,
                "content_length_factor": content_length_factor,
                "title_overlap": title_overlap,
                "content_overlap": content_overlap,
                "summary_overlap": summary_overlap,
                "specificity_bonus": specificity_bonus,
                "threshold_used": dynamic_threshold
            })
            max_relevance = max(max_relevance, combined_relevance)
    
    # Sort by relevance score (highest first)
    relevant_docs.sort(key=lambda x: x["relevance"], reverse=True)
    
    return {
        "is_relevant": len(relevant_docs) > 0,
        "max_relevance": max_relevance,
        "relevant_docs": relevant_docs,
        "total_checked": len(documents),
        "threshold_used": base_threshold,
        "avg_semantic_similarity": sum(doc["semantic_similarity"] for doc in relevant_docs) / len(relevant_docs) if relevant_docs else 0.0
    }
