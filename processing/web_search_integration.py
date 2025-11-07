"""
Web search integration for the core processing pipeline.

Copyright (c) 2025 Mike Tallent & Claude Sonnet - Geniusai.biz
All rights reserved. This software is proprietary and confidential.
"""

import logging
from typing import Dict, Any, Optional
from endpoints.web_search_endpoints import perform_web_search_for_question, enhance_context_with_web_search

logger = logging.getLogger(__name__)

async def integrate_web_search_into_processing(
    question: str, 
    existing_context: str, 
    context: Dict[str, Any] = None,
    update_callback=None
) -> Dict[str, Any]:
    """
    Integrate web search into the question processing pipeline.
    """
    try:
        # Check if web search is enabled
        web_search_enabled = False
        if context:
            web_search_enabled = context.get('webSearchEnabled', False)
            if not web_search_enabled and context.get('settings'):
                web_search_enabled = context['settings'].get('webSearchEnabled', False)
        
        if not web_search_enabled:
            return {
                'enhanced_context': existing_context,
                'web_search_used': False,
                'web_sources': [],
                'message': 'Web search disabled'
            }
        
        # Send update if callback provided
        if update_callback:
            await update_callback('web_search', 30, 'Searching the web for relevant information...')
        
        # Perform web search
        web_search_result = await perform_web_search_for_question(question, context)
        
        if update_callback:
            await update_callback('web_search', 60, 'Processing web search results...')
        
        # Enhance context with web search results
        enhanced_result = await enhance_context_with_web_search(question, existing_context, context)
        
        if update_callback:
            await update_callback('web_search', 100, f"Web search complete - found {enhanced_result.get('total_web_results', 0)} sources")
        
        return enhanced_result
        
    except Exception as e:
        logger.error(f"Web search integration error: {e}")
        return {
            'enhanced_context': existing_context,
            'web_search_used': False,
            'web_sources': [],
            'error': str(e)
        }
