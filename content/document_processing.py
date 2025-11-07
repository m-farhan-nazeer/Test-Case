"""
Document processing utilities for content extraction and chunking.

Copyright (c) 2025 Mike Tallent & Claude Sonnet - Geniusai.biz
All rights reserved. This software is proprietary and confidential.
"""

import logging
import hashlib
from typing import List, Dict, Any
from ai.openai_integration import generate_openai_embedding
from content.extraction_functions import chunk_document_content, calculate_chunk_similarities
from utils.utility_functions import extract_important_terms, calculate_enhanced_overlap
from vector.similarity_functions import calculate_cosine_similarity

logger = logging.getLogger(__name__)


async def extract_best_chunks_for_question(question: str, content: str, summary: str = "", max_length: int = 800, question_type: str = 'general') -> str:
    """Extract the best chunks from content based on question type and relevance."""
    if not content:
        return summary[:max_length] if summary else ""
    
    if len(content) <= max_length:
        return content
    
    # Generate question embedding for similarity comparison
    question_cache_key = f"question_chunks_{hashlib.md5(question.encode()).hexdigest()}"
    question_embedding = await generate_openai_embedding(question, question_cache_key)
    
    # Create chunks with question-type-aware sizing
    if question_type in ['procedural', 'explanatory']:
        chunk_size = 500  # Larger chunks for process/explanation questions
        overlap = 75
    elif question_type in ['definitional', 'personal']:
        chunk_size = 300  # Smaller chunks for specific questions
        overlap = 50
    else:
        chunk_size = 400  # Default size
        overlap = 60
    
    chunks = chunk_document_content(content, chunk_size=chunk_size, overlap=overlap)
    
    if not chunks:
        return content[:max_length] + "..."
    
    # Limit chunks to prevent excessive processing
    max_chunks_to_process = min(15, len(chunks))
    chunks_to_process = chunks[:max_chunks_to_process]
    
    # Calculate enhanced chunk scores
    chunk_scores = []
    for i, chunk in enumerate(chunks_to_process):
        try:
            # Generate embedding for chunk
            chunk_cache_key = f"best_chunk_{i}_{hashlib.md5(chunk.encode()).hexdigest()}"
            chunk_embedding = await generate_openai_embedding(chunk, chunk_cache_key)
            
            # Calculate semantic similarity
            semantic_sim = calculate_cosine_similarity(question_embedding, chunk_embedding)
            
            # Calculate keyword relevance
            question_terms = extract_important_terms(question)
            chunk_words = set(chunk.lower().split())
            keyword_relevance = calculate_enhanced_overlap(question_terms, chunk_words)
            
            # Calculate position score (earlier chunks get slight boost)
            position_score = 1.0 - (i * 0.02)
            
            # Calculate question-type-specific score
            type_score = calculate_chunk_type_relevance(chunk, question_type)
            
            # Combined score
            combined_score = (
                semantic_sim * 0.5 +
                keyword_relevance * 0.3 +
                type_score * 0.15 +
                position_score * 0.05
            )
            
            chunk_scores.append({
                "chunk_index": i,
                "chunk": chunk,
                "score": combined_score,
                "semantic_similarity": semantic_sim,
                "keyword_relevance": keyword_relevance,
                "type_score": type_score
            })
        except Exception as e:
            logger.warning(f"Error processing chunk {i} in extract_best_chunks_for_question: {e}")
            continue
    
    if not chunk_scores:
        return content[:max_length] + "..."
    
    # Sort by score
    chunk_scores.sort(key=lambda x: x["score"], reverse=True)
    
    # Select best chunks that fit within max_length
    selected_chunks = []
    total_length = 0
    
    for chunk_data in chunk_scores:
        chunk = chunk_data["chunk"]
        chunk_length = len(chunk)
        
        if total_length + chunk_length + 10 <= max_length:  # +10 for separators
            selected_chunks.append({
                "chunk": chunk,
                "score": chunk_data["score"],
                "original_index": chunk_data["chunk_index"]
            })
            total_length += chunk_length + 10
        elif total_length == 0:  # If first chunk is too long, truncate it
            truncated = chunk[:max_length-3] + "..."
            selected_chunks.append({
                "chunk": truncated,
                "score": chunk_data["score"],
                "original_index": chunk_data["chunk_index"]
            })
            break
        else:
            break
    
    if not selected_chunks:
        return content[:max_length] + "..."
    
    # Sort selected chunks by their original order for coherent reading
    selected_chunks.sort(key=lambda x: x["original_index"])
    
    # Combine chunks with intelligent transitions
    result_parts = []
    for i, chunk_data in enumerate(selected_chunks):
        chunk = chunk_data["chunk"].strip()
        if i > 0:
            # Add transition if chunks are not consecutive
            prev_index = selected_chunks[i-1]["original_index"]
            curr_index = chunk_data["original_index"]
            if curr_index > prev_index + 1:
                result_parts.append("[...]")
        result_parts.append(chunk)
    
    result = " ".join(result_parts)
    
    # Ensure we don't exceed max_length
    if len(result) > max_length:
        result = result[:max_length-3] + "..."
    
    return result.strip()


def calculate_chunk_type_relevance(chunk: str, question_type: str) -> float:
    """Calculate how well a chunk matches the question type."""
    chunk_lower = chunk.lower()
    
    type_indicators = {
        'definitional': ['definition', 'define', 'means', 'refers to', 'is defined as', 'concept'],
        'procedural': ['step', 'first', 'then', 'next', 'process', 'method', 'procedure', 'how to'],
        'explanatory': ['because', 'reason', 'cause', 'due to', 'result', 'therefore', 'consequently'],
        'temporal': ['when', 'time', 'date', 'year', 'period', 'during', 'before', 'after', 'since'],
        'spatial': ['where', 'location', 'place', 'located', 'situated', 'region', 'area'],
        'personal': ['who', 'person', 'people', 'individual', 'author', 'creator', 'founder'],
        'comparative': ['compare', 'versus', 'difference', 'similar', 'unlike', 'contrast', 'both'],
        'enumerative': ['list', 'examples', 'types', 'kinds', 'categories', 'include', 'such as']
    }
    
    indicators = type_indicators.get(question_type, [])
    if not indicators:
        return 0.5
    
    matches = sum(1 for indicator in indicators if indicator in chunk_lower)
    base_score = min(1.0, matches / len(indicators))
    
    # Boost score if chunk has multiple indicators
    if matches >= 2:
        base_score = min(1.0, base_score * 1.2)
    
    return base_score


async def extract_vectorized_content(question: str, content: str, summary: str = "", max_length: int = 800) -> str:
    """Extract the most relevant content using vectorized similarity matching."""
    if not content:
        return summary[:max_length] if summary else ""
    
    if len(content) <= max_length:
        return content
    
    # Generate question embedding for similarity comparison with caching
    question_cache_key = f"question_extract_{hashlib.md5(question.encode()).hexdigest()}"
    question_embedding = await generate_openai_embedding(question, question_cache_key)
    
    # Split content into chunks for vectorized analysis
    chunks = chunk_document_content(content, chunk_size=400, overlap=50)
    
    if not chunks:
        return content[:max_length] + "..."
    
    # Limit the number of chunks to process to prevent infinite loops
    max_chunks_to_process = min(20, len(chunks))  # Process at most 20 chunks
    chunks_to_process = chunks[:max_chunks_to_process]
    
    logger.info(f"Processing {len(chunks_to_process)} chunks out of {len(chunks)} total chunks for content extraction")
    
    # Calculate vectorized similarities for limited chunks
    chunk_similarities = []
    for i, chunk in enumerate(chunks_to_process):
        try:
            # Generate embedding for chunk with better caching
            chunk_cache_key = f"extract_chunk_{i}_{hashlib.md5(chunk.encode()).hexdigest()}"
            chunk_embedding = await generate_openai_embedding(chunk, chunk_cache_key)
            
            # Calculate semantic similarity
            semantic_sim = calculate_cosine_similarity(question_embedding, chunk_embedding)
            
            chunk_similarities.append({
                "chunk_index": i,
                "chunk": chunk,
                "semantic_similarity": semantic_sim,
                "relevance_score": semantic_sim  # Use semantic similarity as primary relevance
            })
        except Exception as e:
            logger.warning(f"Error processing chunk {i} in extract_vectorized_content: {e}")
            continue
    
    if not chunk_similarities:
        logger.warning("No chunk similarities calculated, falling back to simple truncation")
        return content[:max_length] + "..."
    
    # Sort by semantic similarity
    chunk_similarities.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    # Select the best chunks that fit within max_length
    selected_chunks = []
    total_length = 0
    
    for chunk_data in chunk_similarities:
        chunk = chunk_data["chunk"]
        chunk_length = len(chunk)
        
        # Check if we can fit this chunk
        if total_length + chunk_length + 3 <= max_length:  # +3 for "... "
            selected_chunks.append({
                "chunk": chunk,
                "score": chunk_data["relevance_score"],
                "original_index": chunk_data["chunk_index"]
            })
            total_length += chunk_length + 3
        elif total_length == 0:  # If first chunk is too long, truncate it
            truncated = chunk[:max_length-3] + "..."
            selected_chunks.append({
                "chunk": truncated,
                "score": chunk_data["relevance_score"],
                "original_index": chunk_data["chunk_index"]
            })
            break
        else:
            break
    
    if not selected_chunks:
        return content[:max_length] + "..."
    
    # Sort selected chunks by their original order in the document
    selected_chunks.sort(key=lambda x: x["original_index"])
    
    # Combine chunks with smooth transitions
    excerpt_parts = []
    for i, chunk_data in enumerate(selected_chunks):
        chunk = chunk_data["chunk"].strip()
        if i > 0:
            # Add transition if chunks are not consecutive
            prev_index = selected_chunks[i-1]["original_index"]
            curr_index = chunk_data["original_index"]
            if curr_index > prev_index + 1:
                excerpt_parts.append("...")
        excerpt_parts.append(chunk)
    
    excerpt = " ".join(excerpt_parts)
    
    # Ensure we don't exceed max_length
    if len(excerpt) > max_length:
        excerpt = excerpt[:max_length-3] + "..."
    
    return excerpt.strip()
