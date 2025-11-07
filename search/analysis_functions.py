"""
Question analysis and search functions.

Copyright (c) 2025 Mike Tallent & Claude Sonnet - Geniusai.biz
All rights reserved. This software is proprietary and confidential.
"""

import logging
from typing import List, Dict, Any
from database.database_functions import get_db_connection, execute_db_query

logger = logging.getLogger(__name__)


def analyze_question_type(question_lower: str) -> str:
    """Analyze the type of question to optimize search strategy."""
    if any(word in question_lower for word in ['what', 'define', 'definition', 'meaning']):
        return 'definitional'
    elif any(word in question_lower for word in ['how', 'steps', 'process', 'method', 'way']):
        return 'procedural'
    elif any(word in question_lower for word in ['why', 'reason', 'cause', 'because', 'explain']):
        return 'explanatory'
    elif any(word in question_lower for word in ['when', 'time', 'date', 'year', 'period']):
        return 'temporal'
    elif any(word in question_lower for word in ['where', 'location', 'place', 'country', 'city']):
        return 'spatial'
    elif any(word in question_lower for word in ['who', 'person', 'people', 'author', 'creator']):
        return 'personal'
    elif any(word in question_lower for word in ['compare', 'difference', 'versus', 'vs', 'contrast']):
        return 'comparative'
    elif any(word in question_lower for word in ['list', 'examples', 'types', 'kinds', 'categories']):
        return 'enumerative'
    else:
        return 'general'


def analyze_question_complexity(question: str) -> str:
    """Analyze question complexity to adjust search parameters."""
    word_count = len(question.split())
    
    # Count complex indicators
    complex_indicators = sum([
        1 for word in ['analyze', 'evaluate', 'compare', 'synthesize', 'relationship', 'impact', 'implications']
        if word in question.lower()
    ])
    
    if word_count > 20 or complex_indicators >= 2:
        return 'high'
    elif word_count > 10 or complex_indicators >= 1:
        return 'medium'
    else:
        return 'low'


async def search_documents_in_db(conn, search_terms, limit=10):
    """Search documents in database with consistent logic."""
    cursor = conn.cursor()
    
    if search_terms:
        search_pattern = f'%{search_terms[0]}%'
        cursor.execute("""
            SELECT id, title, content, summary, metadata, embedding
            FROM documents
            WHERE LOWER(content) LIKE ? 
               OR LOWER(title) LIKE ? 
               OR LOWER(summary) LIKE ?
               OR LOWER(metadata) LIKE ?
            ORDER BY 
                CASE 
                    WHEN LOWER(title) LIKE ? THEN 1
                    WHEN LOWER(summary) LIKE ? THEN 2
                    WHEN LOWER(content) LIKE ? THEN 3
                    ELSE 4
                END,
                created_at DESC
            LIMIT ?
        """, (search_pattern, search_pattern, search_pattern, search_pattern,
              search_pattern, search_pattern, search_pattern, limit))
    else:
        # Get recent documents if no search terms
        cursor.execute("SELECT id, title, content, summary, metadata, embedding FROM documents ORDER BY created_at DESC LIMIT ?", (limit,))
    
    results = cursor.fetchall()
    cursor.close()
    return results
