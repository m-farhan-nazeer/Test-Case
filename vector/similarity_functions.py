"""
Vector similarity calculation functions.

Copyright (c) 2025 Mike Tallent & Claude Sonnet - Geniusai.biz
All rights reserved. This software is proprietary and confidential.
"""

import math
import logging
from typing import List

logger = logging.getLogger(__name__)


def calculate_cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Enhanced cosine similarity calculation with improved normalization and scoring."""
    if len(embedding1) != len(embedding2) or len(embedding1) == 0:
        return 0.0
    
    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
    
    # Calculate magnitudes
    magnitude1 = sum(a * a for a in embedding1) ** 0.5
    magnitude2 = sum(b * b for b in embedding2) ** 0.5
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    # Calculate cosine similarity (-1 to 1)
    raw_similarity = dot_product / (magnitude1 * magnitude2)
    
    # Enhanced normalization with better discrimination
    # Use sigmoid-like transformation for better score distribution
    
    # Apply sigmoid transformation to enhance discrimination
    sigmoid_factor = 1 / (1 + math.exp(-10 * (raw_similarity - 0.5)))
    
    # Normalize to 0-1 range
    normalized = (raw_similarity + 1) / 2
    
    # Combine sigmoid and linear normalization for better results
    enhanced_similarity = (normalized * 0.7) + (sigmoid_factor * 0.3)
    
    # Apply dynamic scaling based on similarity strength
    if enhanced_similarity > 0.85:
        # Boost very high similarities
        enhanced_similarity = min(1.0, enhanced_similarity * 1.05)
    elif enhanced_similarity > 0.7:
        # Slightly boost good similarities
        enhanced_similarity = min(1.0, enhanced_similarity * 1.02)
    elif enhanced_similarity < 0.3:
        # Reduce low similarities more aggressively
        enhanced_similarity = max(0.0, enhanced_similarity * 0.8)
    
    return max(0.0, min(1.0, enhanced_similarity))
