"""
Advanced scoring functions for document relevance.

Copyright (c) 2025 Mike Tallent & Claude Sonnet - Geniusai.biz
All rights reserved. This software is proprietary and confidential.
"""

import logging
from typing import Dict, List
from datetime import datetime, timezone
from utils.utility_functions import safe_get_field

logger = logging.getLogger(__name__)


def calculate_advanced_phrase_matching(question: str, doc_fields: Dict[str, str]) -> float:
    """Calculate advanced phrase matching with n-grams."""
    question_lower = question.lower()
    doc_text = f"{doc_fields['title']} {doc_fields['content']} {doc_fields['summary']}".lower()
    
    # Generate n-grams (2-grams, 3-grams, 4-grams)
    phrase_matches = 0
    total_phrases = 0
    
    words = question_lower.split()
    
    # 2-grams
    for i in range(len(words) - 1):
        phrase = f"{words[i]} {words[i+1]}"
        if len(phrase) > 6:  # Skip very short phrases
            total_phrases += 1
            if phrase in doc_text:
                phrase_matches += 2  # Higher weight for 2-grams
    
    # 3-grams
    for i in range(len(words) - 2):
        phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
        total_phrases += 1
        if phrase in doc_text:
            phrase_matches += 3  # Higher weight for 3-grams
    
    # 4-grams
    for i in range(len(words) - 3):
        phrase = f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}"
        total_phrases += 1
        if phrase in doc_text:
            phrase_matches += 4  # Highest weight for 4-grams
    
    return phrase_matches / max(total_phrases, 1)


def assess_content_quality_for_question(doc_fields: Dict[str, str], question_type: str, question_complexity: str) -> float:
    """Assess content quality based on question requirements."""
    content = doc_fields['content']
    title = doc_fields['title']
    summary = doc_fields['summary']
    
    quality_score = 0.0
    
    # Length appropriateness
    content_length = len(content)
    if question_complexity == 'high':
        # Complex questions need detailed content
        if content_length > 2000:
            quality_score += 0.3
        elif content_length > 1000:
            quality_score += 0.2
        else:
            quality_score += 0.1
    elif question_complexity == 'medium':
        if 500 <= content_length <= 3000:
            quality_score += 0.3
        else:
            quality_score += 0.2
    else:  # low complexity
        if content_length > 100:
            quality_score += 0.3
    
    # Structure quality
    if title and len(title.strip()) > 5:
        quality_score += 0.2
    if summary and len(summary.strip()) > 10:
        quality_score += 0.2
    
    # Content structure indicators
    if content.count('\n') > 2:  # Has paragraphs
        quality_score += 0.1
    if content.count('.') > 5:  # Has multiple sentences
        quality_score += 0.1
    if any(indicator in content.lower() for indicator in ['example', 'for instance', 'such as']):
        quality_score += 0.1
    
    return min(1.0, quality_score)


def calculate_freshness_score(doc_fields: Dict[str, str]) -> float:
    """Calculate document freshness score."""
    try:
        created_at = doc_fields.get('created_at', '')
        if not created_at:
            return 0.5  # Neutral score for unknown dates
        
        if isinstance(created_at, str):
            doc_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        else:
            doc_date = created_at
        
        days_old = (datetime.now(timezone.utc) - doc_date).days
        
        # Fresher documents get higher scores
        if days_old <= 1:
            return 1.0
        elif days_old <= 7:
            return 0.9
        elif days_old <= 30:
            return 0.7
        elif days_old <= 90:
            return 0.5
        elif days_old <= 365:
            return 0.3
        else:
            return 0.1
    except:
        return 0.5


def calculate_question_type_match(doc_fields: Dict[str, str], question_type: str) -> float:
    """Calculate how well document matches question type."""
    content = doc_fields['content'].lower()
    title = doc_fields['title'].lower()
    
    type_indicators = {
        'definitional': ['definition', 'define', 'meaning', 'is', 'refers to', 'means'],
        'procedural': ['step', 'process', 'method', 'how to', 'procedure', 'instructions'],
        'explanatory': ['because', 'reason', 'cause', 'explain', 'due to', 'result'],
        'temporal': ['when', 'time', 'date', 'year', 'period', 'during', 'before', 'after'],
        'spatial': ['where', 'location', 'place', 'country', 'city', 'region', 'area'],
        'personal': ['who', 'person', 'people', 'author', 'creator', 'founder', 'leader'],
        'comparative': ['compare', 'versus', 'difference', 'similar', 'unlike', 'contrast'],
        'enumerative': ['list', 'examples', 'types', 'kinds', 'categories', 'include']
    }
    
    indicators = type_indicators.get(question_type, [])
    if not indicators:
        return 0.5
    
    doc_text = f"{title} {content}"
    matches = sum(1 for indicator in indicators if indicator in doc_text)
    
    return min(1.0, matches / len(indicators))


def apply_diversity_filtering(relevant_docs: List[Dict], max_results: int) -> List[Dict]:
    """Apply diversity filtering to avoid too similar results."""
    if len(relevant_docs) <= max_results:
        return relevant_docs
    
    diverse_docs = [relevant_docs[0]]  # Always include the top result
    
    for doc in relevant_docs[1:]:
        if len(diverse_docs) >= max_results:
            break
        
        # Check diversity against already selected docs
        is_diverse = True
        doc_title = safe_get_field(doc["doc"], "title", "").lower()
        doc_content_preview = safe_get_field(doc["doc"], "content", "")[:200].lower()
        
        for selected_doc in diverse_docs:
            selected_title = safe_get_field(selected_doc["doc"], "title", "").lower()
            selected_content_preview = safe_get_field(selected_doc["doc"], "content", "")[:200].lower()
            
            # Simple similarity check
            title_similarity = len(set(doc_title.split()) & set(selected_title.split())) / max(len(doc_title.split()), 1)
            content_similarity = len(set(doc_content_preview.split()) & set(selected_content_preview.split())) / max(len(doc_content_preview.split()), 1)
            
            if title_similarity > 0.7 or content_similarity > 0.8:
                is_diverse = False
                break
        
        if is_diverse:
            diverse_docs.append(doc)
    
    return diverse_docs
