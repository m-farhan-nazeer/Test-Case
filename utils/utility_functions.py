"""
Utility functions for the agentic RAG system.

Copyright (c) 2025 Mike Tallent & Claude Sonnet - Geniusai.biz
All rights reserved. This software is proprietary and confidential.
"""

import json
import logging
from typing import Dict, Any, List
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def safe_get_field(doc, field: str, default: str = "") -> str:
    """Safely get any field from various object types."""
    try:
        if hasattr(doc, 'get'):
            return doc.get(field, default)
        elif hasattr(doc, 'keys'):  # SQLite Row object
            return doc[field] if field in doc.keys() and doc[field] else default
        elif isinstance(doc, dict):
            return doc.get(field, default)
        else:
            return default
    except (KeyError, TypeError, AttributeError):
        return default


def get_document_fields(doc, index=0):
    """Extract common document fields safely."""
    return {
        'id': safe_get_field(doc, 'id', f'doc_{index}'),
        'title': safe_get_field(doc, 'title', 'Untitled'),
        'content': safe_get_field(doc, 'content', ''),
        'summary': safe_get_field(doc, 'summary', ''),
        'metadata': safe_get_field(doc, 'metadata', '{}'),
        'embedding': safe_get_field(doc, 'embedding', ''),
        'created_at': safe_get_field(doc, 'created_at', '')
    }


def parse_document_metadata(metadata_str):
    """Parse metadata string to dict."""
    if isinstance(metadata_str, str):
        try:
            return json.loads(metadata_str)
        except (json.JSONDecodeError, TypeError):
            return {}
    elif isinstance(metadata_str, dict):
        return metadata_str
    return {}


def parse_document_embedding(embedding_str):
    """Parse embedding string to list."""
    if isinstance(embedding_str, str):
        try:
            embedding = json.loads(embedding_str)
            return embedding if isinstance(embedding, list) and len(embedding) > 0 else None
        except (json.JSONDecodeError, TypeError):
            return None
    elif isinstance(embedding_str, list):
        return embedding_str if len(embedding_str) > 0 else None
    return None


def format_personality_context(personality_settings: Dict[str, Any]) -> str:
    """Format personality settings into a context string for the AI."""
    if not personality_settings:
        return ""
    
    personality_traits = []
    
    # Map trait names to more readable descriptions
    trait_descriptions = {
        'openness': 'openness to experience',
        'conscientiousness': 'conscientiousness',
        'extraversion': 'extraversion',
        'agreeableness': 'agreeableness',
        'neuroticism': 'emotional stability',
        'formality': 'formality level',
        'directness': 'directness',
        'politeness': 'politeness',
        'assertiveness': 'assertiveness',
        'humorUsage': 'humor usage',
        'vocalEnergy': 'vocal energy',
        'pacingOfSpeech': 'pacing of speech',
        'clarity': 'clarity',
        'storytelling': 'storytelling ability',
        'questionAsking': 'question asking frequency',
        'empathyLevel': 'empathy level',
        'emotionalIntensity': 'emotional intensity',
        'optimism': 'optimism',
        'patience': 'patience',
        'resilience': 'resilience',
        'gratitudeExpression': 'gratitude expression',
        'forgiveness': 'forgiveness'
    }
    
    for trait, value in personality_settings.items():
        if trait in trait_descriptions and value:
            trait_name = trait_descriptions[trait]
            personality_traits.append(f"{trait_name}: {value.lower()}")
    
    if personality_traits:
        return f"Personality traits to embody: {', '.join(personality_traits[:10])}. " + \
               "Please respond in a way that reflects these personality characteristics while maintaining helpfulness and accuracy."
    
    return ""


def extract_important_terms(question: str) -> List[str]:
    """Extract important terms from question using simple heuristics."""
    import re
    
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
    }
    
    # Extract words, prioritize longer words and capitalized words
    words = re.findall(r'\b[a-zA-Z]+\b', question.lower())
    important_terms = []
    
    for word in words:
        if len(word) > 3 and word not in stop_words:
            # Give higher priority to longer words
            priority = len(word)
            if word in question and word[0].isupper():  # Originally capitalized
                priority += 2
            important_terms.append((word, priority))
    
    # Sort by priority and return top terms
    important_terms.sort(key=lambda x: x[1], reverse=True)
    return [term[0] for term in important_terms[:10]]


def calculate_enhanced_overlap(question_terms: List[str], doc_words: set) -> float:
    """Calculate enhanced overlap with fuzzy matching."""
    if not question_terms:
        return 0.0
    
    matches = 0
    for term in question_terms:
        # Exact match
        if term in doc_words:
            matches += 1
        else:
            # Fuzzy matching for similar words
            for doc_word in doc_words:
                if len(doc_word) > 4 and len(term) > 4:
                    # Simple similarity check
                    if abs(len(doc_word) - len(term)) <= 2:
                        common_chars = set(term) & set(doc_word)
                        if len(common_chars) / max(len(term), len(doc_word)) > 0.7:
                            matches += 0.5
                            break
    
    return matches / len(question_terms)


def get_keyword_weights_by_question_type(question_type: str) -> Dict[str, float]:
    """Get keyword weights based on question type."""
    weights = {
        'definitional': {'title': 0.6, 'content': 0.3, 'summary': 0.1},
        'procedural': {'title': 0.3, 'content': 0.6, 'summary': 0.1},
        'explanatory': {'title': 0.4, 'content': 0.5, 'summary': 0.1},
        'temporal': {'title': 0.5, 'content': 0.4, 'summary': 0.1},
        'spatial': {'title': 0.5, 'content': 0.4, 'summary': 0.1},
        'personal': {'title': 0.6, 'content': 0.3, 'summary': 0.1},
        'comparative': {'title': 0.4, 'content': 0.5, 'summary': 0.1},
        'enumerative': {'title': 0.3, 'content': 0.6, 'summary': 0.1},
        'general': {'title': 0.5, 'content': 0.3, 'summary': 0.2}
    }
    return weights.get(question_type, weights['general'])
