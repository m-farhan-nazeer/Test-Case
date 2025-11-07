"""
Dynamic evaluation functions for adaptive scoring.

Copyright (c) 2025 Mike Tallent & Claude Sonnet - Geniusai.biz
All rights reserved. This software is proprietary and confidential.
"""

import logging
import hashlib
from typing import Dict, List, Any
from ai.openai_integration import generate_openai_embedding
from vector.similarity_functions import calculate_cosine_similarity
from content.extraction_functions import chunk_document_content

logger = logging.getLogger(__name__)


def adjust_weights_for_question(base_weights: Dict[str, float], question_type: str, question_complexity: str) -> Dict[str, float]:
    """Adjust weights based on question characteristics."""
    adjusted = base_weights.copy()
    
    # Adjust for question type
    if question_type == 'definitional':
        adjusted['semantic'] += 0.1
        adjusted['keyword'] -= 0.05
        adjusted['phrase'] -= 0.05
    elif question_type == 'procedural':
        adjusted['content_quality'] += 0.1
        adjusted['chunk_relevance'] += 0.05
        adjusted['semantic'] -= 0.15
    elif question_type == 'comparative':
        adjusted['chunk_relevance'] += 0.1
        adjusted['content_quality'] += 0.05
        adjusted['semantic'] -= 0.15
    
    # Adjust for complexity
    if question_complexity == 'high':
        adjusted['content_quality'] += 0.1
        adjusted['chunk_relevance'] += 0.05
        adjusted['keyword'] -= 0.15
    elif question_complexity == 'low':
        adjusted['keyword'] += 0.1
        adjusted['phrase'] += 0.05
        adjusted['content_quality'] -= 0.15
    
    # Normalize weights to sum to 1.0
    total_weight = sum(adjusted.values())
    for key in adjusted:
        adjusted[key] = adjusted[key] / total_weight
    
    return adjusted


def calculate_dynamic_threshold(base_threshold: float, semantic_similarity: float, keyword_score: float, question_type: str, question_complexity: str) -> float:
    """Calculate dynamic threshold with enhanced criteria for excellent quality."""
    threshold = base_threshold
    
    # Enhanced adjustments based on semantic similarity
    if semantic_similarity >= 0.9:
        threshold = max(0.25, threshold - 0.2)   # Premium threshold for excellent semantic match
    elif semantic_similarity >= 0.8:
        threshold = max(0.2, threshold - 0.15)   # High quality threshold
    elif semantic_similarity >= 0.7:
        threshold = max(0.25, threshold - 0.1)   # Good quality threshold
    elif semantic_similarity >= 0.6:
        threshold = max(0.3, threshold - 0.05)   # Moderate quality threshold
    elif semantic_similarity < 0.4:
        threshold = threshold + 0.15             # Raise threshold for poor semantic match
    
    # Enhanced adjustments based on keyword score
    if keyword_score >= 0.8:
        threshold = max(0.25, threshold - 0.08)  # Excellent keyword match
    elif keyword_score >= 0.7:
        threshold = max(0.3, threshold - 0.05)   # Good keyword match
    elif keyword_score < 0.3:
        threshold = threshold + 0.1              # Poor keyword match
    elif keyword_score < 0.2:
        threshold = threshold + 0.15             # Very poor keyword match
    
    # Enhanced adjustments based on question type
    if question_type in ['definitional', 'personal']:
        threshold = max(0.3, threshold - 0.05)   # Specific questions need good matches
    elif question_type in ['comparative', 'explanatory']:
        threshold = threshold + 0.08             # Complex questions need excellent matches
    elif question_type in ['procedural']:
        threshold = threshold + 0.05             # Process questions need detailed matches
    
    # Enhanced adjustments based on complexity
    if question_complexity == 'high':
        threshold = threshold + 0.1              # High complexity requires excellent matches
    elif question_complexity == 'medium':
        threshold = threshold + 0.05             # Medium complexity requires good matches
    elif question_complexity == 'low':
        threshold = max(0.25, threshold - 0.03)  # Simple questions can have lower threshold
    
    # Ensure threshold promotes high quality results
    return min(0.85, max(0.2, threshold))


def evaluate_document_relevance(combined_relevance: float, semantic_similarity: float, keyword_score: float, phrase_score: float, threshold: float, question_type: str) -> bool:
    """Enhanced relevance evaluation with multiple criteria."""
    # Basic threshold check
    if combined_relevance < threshold:
        return False
    
    # Additional quality checks
    if semantic_similarity >= 0.7:
        return True  # High semantic similarity is always good
    
    if keyword_score >= 0.6 and phrase_score >= 0.3:
        return True  # Good keyword and phrase matching
    
    if question_type in ['definitional', 'personal'] and semantic_similarity >= 0.5:
        return True  # Lower bar for specific question types
    
    if combined_relevance >= threshold * 1.2:
        return True  # Significantly above threshold
    
    # Minimum quality requirements
    return (
        semantic_similarity >= 0.3 or 
        keyword_score >= 0.4 or 
        phrase_score >= 0.3
    )


async def calculate_chunk_relevance_score(question: str, question_embedding: List[float], content: str) -> float:
    """Calculate relevance score based on best matching chunks."""
    try:
        chunks = chunk_document_content(content, chunk_size=300, overlap=30)
        if not chunks:
            return 0.0
        
        # Limit chunks to prevent excessive processing
        max_chunks = min(5, len(chunks))  # Reduced from 10 to 5
        best_similarity = 0.0
        
        for i, chunk in enumerate(chunks[:max_chunks]):
            chunk_cache_key = f"chunk_rel_{i}_{hashlib.md5(chunk.encode()).hexdigest()}"
            chunk_embedding = await generate_openai_embedding(chunk, chunk_cache_key)
            similarity = calculate_cosine_similarity(question_embedding, chunk_embedding)
            best_similarity = max(best_similarity, similarity)
        
        return best_similarity
    except Exception as e:
        logger.warning(f"Error calculating chunk relevance: {e}")
        return 0.0
