"""
Content extraction and processing functions.

Copyright (c) 2025 Mike Tallent & Claude Sonnet - Geniusai.biz
All rights reserved. This software is proprietary and confidential.
"""

import hashlib
import logging
from typing import List, Dict, Any
from ai.openai_integration import generate_openai_embedding
from vector.similarity_functions import calculate_cosine_similarity

logger = logging.getLogger(__name__)


def chunk_document_content(content: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split document content into overlapping chunks for better similarity matching."""
    if len(content) <= chunk_size:
        return [content]
    
    chunks = []
    start = 0
    
    while start < len(content):
        end = start + chunk_size
        
        # Try to break at sentence boundaries
        if end < len(content):
            # Look for sentence endings within the last 100 characters
            sentence_end = -1
            for i in range(max(0, end - 100), end):
                if content[i] in '.!?':
                    sentence_end = i + 1
                    break
            
            if sentence_end > start + chunk_size // 2:  # Only use if it's not too short
                end = sentence_end
        
        chunk = content[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        
        # Prevent infinite loop
        if start >= len(content):
            break
    
    return chunks


async def calculate_chunk_similarities(question: str, document_chunks: List[str], question_embedding: List[float]) -> List[Dict[str, Any]]:
    """Calculate similarity scores for document chunks against the question."""
    chunk_similarities = []
    
    # Limit the number of chunks to prevent infinite loops
    max_chunks_to_process = min(8, len(document_chunks))  # Reduced from 15 to 8
    chunks_to_process = document_chunks[:max_chunks_to_process]
    
    logger.info(f"Processing {len(chunks_to_process)} chunks out of {len(document_chunks)} total chunks for similarity calculation")
    
    for i, chunk in enumerate(chunks_to_process):
        try:
            # Generate embedding for chunk with better caching
            chunk_cache_key = f"similarity_chunk_{i}_{hashlib.md5(chunk.encode()).hexdigest()}"
            chunk_embedding = await generate_openai_embedding(chunk, chunk_cache_key)
            
            # Calculate semantic similarity
            semantic_sim = calculate_cosine_similarity(question_embedding, chunk_embedding)
            
            # Calculate keyword overlap
            question_words = set(question.lower().split())
            chunk_words = set(chunk.lower().split())
            keyword_overlap = len(question_words.intersection(chunk_words)) / max(len(question_words), 1)
            
            # Calculate phrase matching
            question_phrases = []
            question_tokens = question.lower().split()
            for j in range(len(question_tokens) - 1):
                question_phrases.append(f"{question_tokens[j]} {question_tokens[j+1]}")
            
            phrase_matches = sum(1 for phrase in question_phrases if phrase in chunk.lower())
            phrase_score = phrase_matches / max(len(question_phrases), 1)
            
            # Combined relevance score
            combined_score = (
                semantic_sim * 0.6 +
                keyword_overlap * 0.25 +
                phrase_score * 0.15
            )
            
            chunk_similarities.append({
                "chunk_index": i,
                "chunk": chunk,
                "semantic_similarity": semantic_sim,
                "keyword_overlap": keyword_overlap,
                "phrase_score": phrase_score,
                "combined_score": combined_score
            })
        except Exception as e:
            logger.warning(f"Error processing chunk {i} in calculate_chunk_similarities: {e}")
            continue
    
    # Sort by combined score
    chunk_similarities.sort(key=lambda x: x["combined_score"], reverse=True)
    return chunk_similarities


async def extract_most_relevant_excerpt(question: str, content: str, summary: str = "", max_length: int = 500) -> str:
    """Extract the most relevant excerpt from document content using advanced chunking and similarity."""
    if not content:
        return summary[:max_length] if summary else ""
    
    # If content is short enough, return it all
    if len(content) <= max_length:
        return content
    
    # Generate question embedding for similarity comparison
    question_embedding = await generate_openai_embedding(question, f"question_{hash(question)}")
    
    # Split content into chunks for better analysis
    chunks = chunk_document_content(content, chunk_size=300, overlap=50)
    
    if not chunks:
        return content[:max_length] + "..."
    
    # Calculate similarities for all chunks
    chunk_similarities = await calculate_chunk_similarities(question, chunks, question_embedding)
    
    # Get the best chunks that fit within max_length
    selected_chunks = []
    total_length = 0
    
    for chunk_data in chunk_similarities:
        chunk = chunk_data["chunk"]
        chunk_length = len(chunk)
        
        # Check if we can fit this chunk
        if total_length + chunk_length + 3 <= max_length:  # +3 for "... "
            selected_chunks.append({
                "chunk": chunk,
                "score": chunk_data["combined_score"],
                "original_index": chunk_data["chunk_index"]
            })
            total_length += chunk_length + 3
        elif total_length == 0:  # If first chunk is too long, truncate it
            truncated = chunk[:max_length-3] + "..."
            selected_chunks.append({
                "chunk": truncated,
                "score": chunk_data["combined_score"],
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
