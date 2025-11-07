"""
Context creation functions for AI responses.

Copyright (c) 2025 Mike Tallent & Claude Sonnet - Geniusai.biz
All rights reserved. This software is proprietary and confidential.
"""

import logging
import hashlib
from typing import List, Dict, Any
from search.analysis_functions import analyze_question_type, analyze_question_complexity
from content.extraction_functions import extract_most_relevant_excerpt, chunk_document_content, calculate_chunk_similarities
from ai.openai_integration import generate_openai_embedding
from vector.similarity_functions import calculate_cosine_similarity
from utils.utility_functions import safe_get_field, get_document_fields, extract_important_terms

logger = logging.getLogger(__name__)


async def create_vectorized_context(question: str, relevant_docs: List[Dict], max_context_length: int = 4000) -> str:
    """Create optimized context from enhanced vectorized search results with intelligent content selection."""
    if not relevant_docs:
        return ""
    
    context_parts = []
    current_length = 0
    
    # Analyze question to determine optimal context strategy
    question_type = analyze_question_type(question.lower())
    question_complexity = analyze_question_complexity(question)
    
    # Adjust number of documents based on question complexity
    if question_complexity == 'high':
        max_docs_to_process = min(6, len(relevant_docs))
    elif question_complexity == 'medium':
        max_docs_to_process = min(4, len(relevant_docs))
    else:
        max_docs_to_process = min(3, len(relevant_docs))
    
    docs_to_process = relevant_docs[:max_docs_to_process]
    
    logger.info(f"Creating enhanced vectorized context from {len(docs_to_process)} documents (question type: {question_type}, complexity: {question_complexity})")
    
    # Calculate dynamic content allocation per document
    base_allocation = max_context_length // max(len(docs_to_process), 1)
    
    for i, doc_data in enumerate(docs_to_process):
        doc = doc_data["doc"]
        doc_fields = get_document_fields(doc, i)
        
        # Dynamic content length based on document relevance and position
        relevance_multiplier = 1.0 + (doc_data['relevance'] - 0.5)  # Boost high-relevance docs
        position_multiplier = 1.0 - (i * 0.1)  # Slightly favor earlier documents
        
        dynamic_allocation = int(base_allocation * relevance_multiplier * position_multiplier)
        max_content_length = min(dynamic_allocation, max_context_length - current_length - 200)
        
        if max_content_length < 100:  # Skip if too little space left
            break
        
        content_length = len(doc_fields['content'])
        
        # Enhanced content extraction strategy
        if content_length > 3000:  # Very large documents
            logger.info(f"Using chunk-based extraction for large document: {doc_fields['title']} ({content_length} chars)")
            relevant_content = await extract_best_chunks_for_question(
                question, doc_fields['content'], doc_fields['summary'], 
                max_length=max_content_length, question_type=question_type
            )
        elif content_length > 1000:  # Medium documents
            try:
                relevant_content = await extract_vectorized_content(
                    question, doc_fields['content'], doc_fields['summary'],
                    max_length=max_content_length
                )
            except Exception as e:
                logger.warning(f"Error in vectorized content extraction for doc {i}, using chunk-based fallback: {e}")
                relevant_content = await extract_best_chunks_for_question(
                    question, doc_fields['content'], doc_fields['summary'],
                    max_length=max_content_length, question_type=question_type
                )
        else:  # Small documents
            relevant_content = doc_fields['content'] if content_length <= max_content_length else doc_fields['content'][:max_content_length] + "..."
        
        if not relevant_content:
            continue
        
        # Enhanced context entry with more detailed relevance information
        quality_indicators = []
        if doc_data.get('content_quality_score', 0) > 0.7:
            quality_indicators.append("High Quality")
        if doc_data.get('chunk_relevance_score', 0) > 0.6:
            quality_indicators.append("Highly Relevant Sections")
        if doc_data.get('question_type_match', 0) > 0.6:
            quality_indicators.append(f"Good {question_type.title()} Match")
        
        quality_info = f" [{', '.join(quality_indicators)}]" if quality_indicators else ""
        
        context_entry = (
            f"Document '{doc_fields['title']}' "
            f"(Relevance: {doc_data['relevance']:.3f}, "
            f"Semantic: {doc_data['semantic_similarity']:.3f}){quality_info}: "
            f"{relevant_content}"
        )
        
        # Check if adding this entry would exceed the limit
        if current_length + len(context_entry) > max_context_length:
            if i == 0:  # Always include at least the first document, even if truncated
                available_space = max_context_length - current_length - 3
                if available_space > 100:  # Only truncate if we have reasonable space
                    truncated_entry = context_entry[:available_space] + "..."
                    context_parts.append(truncated_entry)
            break
        
        context_parts.append(context_entry)
        current_length += len(context_entry) + 2  # +2 for separator
        
        # Dynamic stopping condition based on context quality
        context_density = current_length / max_context_length
        if context_density >= 0.85 and i >= 1:  # Stop if we have good coverage and at least 2 docs
            break
    
    # Add context summary for complex questions
    if question_complexity == 'high' and len(context_parts) > 2:
        summary_info = f"\n[Context Summary: {len(context_parts)} documents covering {question_type} question, total length: {current_length} chars]"
        if current_length + len(summary_info) <= max_context_length:
            context_parts.append(summary_info)
    
    return "\n\n".join(context_parts)


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
            from utils.utility_functions import calculate_enhanced_overlap
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
