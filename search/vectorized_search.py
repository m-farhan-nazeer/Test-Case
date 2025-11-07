"""
Vectorized search functions for document retrieval.

Copyright (c) 2025 Mike Tallent & Claude Sonnet - Geniusai.biz
All rights reserved. This software is proprietary and confidential.
"""

import logging
import hashlib
from typing import List, Dict, Any
from datetime import datetime, timezone
from ai.openai_integration import generate_openai_embedding
from vector.similarity_functions import calculate_cosine_similarity
from utils.utility_functions import (
    safe_get_field, get_document_fields, parse_document_metadata, parse_document_embedding,
    extract_important_terms, calculate_enhanced_overlap, get_keyword_weights_by_question_type
)
from search.analysis_functions import analyze_question_type, analyze_question_complexity
from scoring.advanced_scoring import (
    calculate_advanced_phrase_matching, assess_content_quality_for_question,
    calculate_freshness_score, calculate_question_type_match, apply_diversity_filtering
)
from scoring.dynamic_evaluation import (
    adjust_weights_for_question, calculate_dynamic_threshold, evaluate_document_relevance,
    calculate_chunk_relevance_score
)

logger = logging.getLogger(__name__)


async def perform_vectorized_search(question: str, question_embedding: List[float], documents: List[Dict], relevance_threshold: float = 0.25, max_results: int = 5) -> Dict[str, Any]:
    """Perform vectorized search using embeddings with enhanced relevance scoring."""
    if not documents:
        return {
            "relevant_docs": [],
            "max_relevance": 0.0,
            "total_checked": 0,
            "search_method": "vectorized",
            "threshold_used": relevance_threshold,
            "avg_semantic_similarity": 0.0
        }
    
    logger.info(f"Starting vectorized search with {len(documents)} documents, threshold: {relevance_threshold}")
    
    relevant_docs = []
    max_relevance = 0.0
    semantic_similarities = []
    
    # Extract question keywords for enhanced matching
    question_terms = extract_important_terms(question)
    question_keywords = set(word.lower() for word in question_terms if len(word) > 2)
    
    for i, doc in enumerate(documents):
        try:
            # Extract document fields safely using the utility function
            doc_fields = get_document_fields(doc, i)
            doc_content = doc_fields['content']
            doc_title = doc_fields['title']
            doc_summary = doc_fields['summary']
            doc_id = str(doc_fields['id'])
            
            # Get document embedding
            doc_embedding = None
            if isinstance(doc, dict) and 'embedding' in doc:
                doc_embedding_raw = doc['embedding']
                if isinstance(doc_embedding_raw, str):
                    try:
                        import json
                        doc_embedding = json.loads(doc_embedding_raw)
                    except json.JSONDecodeError:
                        doc_embedding = None
                elif isinstance(doc_embedding_raw, list):
                    doc_embedding = doc_embedding_raw
            
            # Generate embedding if not available
            if not doc_embedding or not isinstance(doc_embedding, list):
                doc_text = f"{doc_title} {doc_content} {doc_summary}"
                if doc_text.strip():
                    doc_cache_key = f"doc_search_{doc_id}_{hashlib.md5(doc_text.encode()).hexdigest()}"
                    doc_embedding = await generate_openai_embedding(doc_text, doc_cache_key)
                else:
                    logger.warning(f"Document {doc_id} has no content for embedding")
                    continue
            
            if not doc_embedding or len(doc_embedding) == 0:
                logger.warning(f"Failed to get embedding for document {doc_id}")
                continue
            
            # Calculate semantic similarity
            semantic_similarity = calculate_cosine_similarity(question_embedding, doc_embedding)
            semantic_similarities.append(semantic_similarity)
            
            # Calculate keyword relevance
            doc_words = set((doc_title + " " + doc_content + " " + doc_summary).lower().split())
            keyword_overlap = calculate_enhanced_overlap(question_keywords, doc_words)
            
            # Calculate title relevance (higher weight)
            title_words = set(doc_title.lower().split())
            title_relevance = len(question_keywords.intersection(title_words)) / max(len(question_keywords), 1)
            
            # Calculate phrase matching
            question_lower = question.lower()
            doc_text_lower = (doc_title + " " + doc_content + " " + doc_summary).lower()
            
            # Simple phrase matching
            question_phrases = []
            question_tokens = question_lower.split()
            for j in range(len(question_tokens) - 1):
                phrase = f"{question_tokens[j]} {question_tokens[j+1]}"
                question_phrases.append(phrase)
            
            phrase_matches = sum(1 for phrase in question_phrases if phrase in doc_text_lower)
            phrase_score = phrase_matches / max(len(question_phrases), 1) if question_phrases else 0
            
            # Content quality factor
            content_quality = min(1.0, len(doc_content) / 1000) if doc_content else 0
            
            # Combined relevance score with balanced weighting
            combined_relevance = (
                semantic_similarity * 0.5 +      # Semantic understanding
                keyword_overlap * 0.2 +          # Keyword matching
                title_relevance * 0.15 +         # Title relevance
                phrase_score * 0.1 +             # Phrase matching
                content_quality * 0.05           # Content quality
            )
            
            logger.debug(f"Document {doc_id}: semantic={semantic_similarity:.3f}, keyword={keyword_overlap:.3f}, title={title_relevance:.3f}, combined={combined_relevance:.3f}")
            
            # Use a more lenient threshold for document relevance
            if combined_relevance >= relevance_threshold or semantic_similarity >= 0.3:
                relevant_docs.append({
                    "doc": doc,
                    "relevance": combined_relevance,
                    "semantic_similarity": semantic_similarity,
                    "keyword_score": keyword_overlap,
                    "title_relevance": title_relevance,
                    "phrase_score": phrase_score,
                    "content_quality": content_quality,
                    "doc_id": doc_id,
                    "doc_title": doc_title[:100] + "..." if len(doc_title) > 100 else doc_title
                })
                max_relevance = max(max_relevance, combined_relevance)
        
        except Exception as e:
            logger.error(f"Error processing document {i} in vectorized search: {e}")
            continue
    
    # Sort by relevance score
    relevant_docs.sort(key=lambda x: x["relevance"], reverse=True)
    
    # Limit results
    if len(relevant_docs) > max_results:
        relevant_docs = relevant_docs[:max_results]
    
    avg_semantic_similarity = sum(semantic_similarities) / len(semantic_similarities) if semantic_similarities else 0.0
    
    logger.info(f"Vectorized search completed: {len(relevant_docs)} relevant docs found out of {len(documents)} total")
    logger.info(f"Max relevance: {max_relevance:.3f}, Avg semantic similarity: {avg_semantic_similarity:.3f}")
    
    if relevant_docs:
        logger.info(f"Top relevant document: '{relevant_docs[0]['doc_title']}' (relevance: {relevant_docs[0]['relevance']:.3f})")
    
    return {
        "relevant_docs": relevant_docs,
        "max_relevance": max_relevance,
        "total_checked": len(documents),
        "search_method": "vectorized",
        "threshold_used": relevance_threshold,
        "avg_semantic_similarity": avg_semantic_similarity
    }


