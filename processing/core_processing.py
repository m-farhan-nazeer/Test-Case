"""
Core question processing logic for the agentic RAG system.

Copyright (c) 2025 Mike Tallent & Claude Sonnet - Geniusai.biz
All rights reserved. This software is proprietary and confidential.
"""

import logging
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List
from ai.openai_integration import generate_openai_answer, generate_openai_embedding, embedding_cache, OPENAI_EMBEDDING_MODEL
from core.database import milvus_store, milvus_user_manager
from database.milvus_functions import ensure_milvus_connection
from search.vectorized_search import perform_vectorized_search
from context.context_creation import create_vectorized_context
from context.formatting_functions import get_user_context
from utils.utility_functions import format_personality_context, get_document_fields
from synthesis.synthesis_agent import SynthesisAgent
from ai.openai_integration import openai_client
from endpoints.web_search_endpoints import perform_web_search_for_question, enhance_context_with_web_search

logger = logging.getLogger(__name__)

# Initialize synthesis agent
synthesis_agent = SynthesisAgent(openai_client)

# Global variables - these will be set by main.py
in_memory_documents = []
system_stats = {}

def set_global_variables(memory_docs, stats):
    """Set global variables from main.py"""
    global in_memory_documents, system_stats
    in_memory_documents = memory_docs
    system_stats = stats


async def process_question_unified(question: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Unified question processing logic for both HTTP and WebSocket endpoints."""
    return await process_question_unified_with_updates(question, context)


async def process_question_unified_with_updates(
    question: str, 
    context: Dict[str, Any] = None, 
    websocket=None, 
    update_callback=None
) -> Dict[str, Any]:
    """Unified question processing logic with real-time updates."""
    start_time = datetime.now(timezone.utc)
    
    async def send_update(step: str, progress: int, message: str):
        """Send processing update if callback is available."""
        if update_callback and websocket:
            await update_callback(step, progress, message, websocket)
        logger.info(f"Processing Step: {step} ({progress}%) - {message}")
    
    # Clear embedding cache at the start of each request
    global embedding_cache
    embedding_cache.clear()
    
    await send_update("analyzing", 10, "Analyzing your question and determining the best approach...")
    
    # Check for URLs in the question first
    from utils.url_detection import analyze_query_for_urls
    url_analysis = analyze_query_for_urls(question)
    
    # Get user context and merge with request context first for faster processing
    user_context = await get_user_context()
    merged_context = {**(context or {}), **user_context}
    
    # Extract personality settings from context
    personality_settings = merged_context.get('personality', {})
    personality_context = format_personality_context(personality_settings)
    
    # Initialize web content for context
    web_content = ""
    web_sources = []
    
    # Handle URL scraping if URLs are detected
    if url_analysis["has_urls"]:
        await send_update("web_search", 25, f"Detected {url_analysis['url_count']} URL(s). Scraping web content...")
        
        from web.scraping_helpers import scrape_website_content
        
        for url_info in url_analysis["urls"]:
            if url_info["valid"]:
                try:
                    logger.info(f"Scraping URL: {url_info['normalized']}")
                    scraped_data = await scrape_website_content(url_info["normalized"])
                    
                    if scraped_data and scraped_data.get("content"):
                        web_content += f"\n\n--- Content from {url_info['domain']} ---\n"
                        web_content += f"Title: {scraped_data.get('title', 'No title')}\n"
                        web_content += f"URL: {url_info['normalized']}\n"
                        web_content += f"Content: {scraped_data['content'][:3000]}...\n"  # Limit content length
                        
                        web_sources.append({
                            "url": url_info["normalized"],
                            "domain": url_info["domain"],
                            "title": scraped_data.get("title", "No title"),
                            "content_length": len(scraped_data["content"]),
                            "scraped_successfully": True
                        })
                        
                        logger.info(f"Successfully scraped {len(scraped_data['content'])} characters from {url_info['domain']}")
                    else:
                        logger.warning(f"No content scraped from {url_info['normalized']}")
                        web_sources.append({
                            "url": url_info["normalized"],
                            "domain": url_info["domain"],
                            "title": "Failed to scrape",
                            "content_length": 0,
                            "scraped_successfully": False
                        })
                        
                except Exception as e:
                    logger.error(f"Error scraping URL {url_info['normalized']}: {e}")
                    web_sources.append({
                        "url": url_info["normalized"],
                        "domain": url_info["domain"],
                        "title": f"Error: {str(e)}",
                        "content_length": 0,
                        "scraped_successfully": False
                    })
        
        if web_content:
            await send_update("web_search", 35, f"Successfully scraped content from {len([s for s in web_sources if s['scraped_successfully']])} website(s)")
            # Use the cleaned query without URLs for document search
            question = url_analysis["query_without_urls"] or question
        else:
            await send_update("web_search", 35, "Web scraping completed but no content retrieved")
    
    await send_update("analyzing", 40, "Checking available knowledge sources...")
    
    # Check if web search is enabled
    web_search_enabled = False
    web_search_used = False
    web_sources = []
    web_content = ""
    
    if context:
        web_search_enabled = context.get('webSearchEnabled', False)
        if not web_search_enabled and context.get('settings'):
            web_search_enabled = context['settings'].get('webSearchEnabled', False)
    
    # Perform web search if enabled
    if web_search_enabled:
        await send_update("web_search", 25, "Performing web search for additional information...")
        
        try:
            web_search_result = await perform_web_search_for_question(question, context)
            
            if web_search_result.get('web_search_enabled') and web_search_result.get('web_results'):
                web_search_used = True
                web_sources = web_search_result.get('web_results', [])
                web_content = web_search_result.get('web_content', '')
                
                await send_update("web_search", 35, f"Found {len(web_sources)} relevant web sources")
                logger.info(f"Web search completed: found {len(web_sources)} web sources")
            else:
                await send_update("web_search", 35, "Web search completed but no relevant results found")
                logger.info("Web search was enabled but no results found")
                
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            await send_update("web_search", 35, "Web search encountered an error")
    
    # Check if we have documents in Milvus or in memory
    documents_available = False
    all_docs = []
    
    # Try to get documents from Milvus first
    if await ensure_milvus_connection():
        try:
            all_docs = await milvus_store.list_all_documents()
            if all_docs:
                documents_available = True
                logger.info(f"Retrieved {len(all_docs)} documents from Milvus for vectorized search")
                await send_update("vector_search", 30, f"Found {len(all_docs)} documents in knowledge base...")
        except Exception as e:
            logger.error(f"Milvus query error: {e}")
    
    # Check in-memory documents if no Milvus documents
    if not documents_available and in_memory_documents:
        documents_available = True
        all_docs = in_memory_documents
        logger.info(f"Using {len(all_docs)} in-memory documents for vectorized search")
        await send_update("vector_search", 30, f"Using {len(all_docs)} in-memory documents...")
    
    if not documents_available:
        await send_update("ai_generation", 60, "No documents found. Generating response using web content and general AI knowledge...")
        
        # No documents available - provide direct AI response with personality and web content
        user_greeting = f"Hello {user_context.get('user_name', 'there')}! " if user_context.get('user_name') else ""
        context_prompt = f"{user_greeting}"
        
        if user_context.get('personalization_note'):
            context_prompt += f"User context: {user_context['personalization_note']} "
        
        if personality_context:
            context_prompt += f"{personality_context} "
        
        # Add web content if available
        if web_search_used and web_content:
            context_prompt += f"\n\nWeb search results:\n{web_content}\n\n"
            context_prompt += "Please provide a comprehensive response based on the web search results above and your general knowledge. Reference specific information from the web sources when relevant."
        else:
            context_prompt += "Please provide a helpful, accurate, and personalized response based on your general knowledge while maintaining the specified personality traits and communication style."
        
        await send_update("ai_generation", 80, "Generating comprehensive AI response...")
        
        # Generate AI response directly without document search but with web content
        openai_response = await generate_openai_answer(question, context_prompt)
        
        await send_update("ai_generation", 100, "Response generated successfully!")
        
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        # Update system statistics
        system_stats["questions_processed"] += 1
        system_stats["total_response_time"] += processing_time
        system_stats["cache_misses"] += 1
        
        # Use higher confidence for general knowledge responses
        general_confidence = {
            "confidence": 0.85,  # 85% confidence for general queries
            "confidence_factors": {
                "general_knowledge": 0.85,
                "response_quality": 0.8,
                "personality_alignment": 0.9
            },
            "confidence_level": "high",
            "reasoning": "General knowledge response with high confidence"
        }
        
        # Include web sources in response if available
        sources_used = []
        citations = []
        reasoning_suffix = ""
        
        if web_search_used and web_sources:
            sources_used = [f"{source['title']} ({source['url']})" for source in web_sources]
            citations = [
                {
                    "source": source["url"],
                    "title": source["title"],
                    "content": f"Web search result: {source.get('snippet', 'No snippet available')}",
                    "confidence": 0.9,
                    "text": f"Web content from {source['title']}"
                }
                for source in web_sources
            ]
            reasoning_suffix = f" Used web search agent to gather information from {len(web_sources)} web source(s)."
        
        response_data = {
            "answer": openai_response["answer"],
            "confidence": general_confidence["confidence"],
            "confidence_factors": general_confidence["confidence_factors"],
            "confidence_level": general_confidence["confidence_level"],
            "sources_used": sources_used,
            "citations": citations,
            "reasoning": f"General knowledge response with web content.{reasoning_suffix} {general_confidence['reasoning']}",
            "processing_time": round(processing_time / 1000, 3),
            "processing_time_ms": int(processing_time),
            "api_used": "openai",
            "vector_search_enabled": False,
            "embedding_model": OPENAI_EMBEDDING_MODEL,
            "results_count": 0,
            "tokens_used": openai_response["tokens_used"],
            "web_search_enabled": web_search_enabled,
            "web_search_used": web_search_used,
            "web_sources": web_sources if web_search_used else []
        }
        
        # Store chat history in Milvus
        try:
            if user_context.get('user_id'):
                await milvus_user_manager.create_chat_entry(
                    user_id=user_context['user_id'],
                    question=question,
                    answer=openai_response["answer"],
                    confidence=general_confidence["confidence"],
                    sources_used=[],
                    reasoning=response_data["reasoning"],
                    processing_time=processing_time
                )
        except Exception as e:
            logger.warning(f"Failed to store chat history in Milvus: {e}")
        
        return response_data
    
    # Documents are available - check for quick general queries first
    # Skip document search for very general queries to provide faster responses
    general_query_keywords = [
        'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
        'how are you', 'what can you do', 'help me', 'tell me about yourself',
        'what is your name', 'who are you', 'what are your capabilities',
        'thank you', 'thanks', 'goodbye', 'bye', 'see you later'
    ]
    
    question_lower = question.lower().strip()
    # Make general query detection much more restrictive - only for actual greetings
    is_basic_greeting = (
        question_lower in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'how are you', 'what can you do', 'who are you', 'thank you', 'thanks', 'goodbye', 'bye'] or
        question_lower.startswith(('hello', 'hi ', 'hey ')) and len(question.split()) <= 2
    )
    
    if is_basic_greeting:
        await send_update("ai_generation", 60, "Detected greeting. Generating personalized response...")
        
        # Provide direct AI response for general queries without document search
        user_greeting = f"Hello {user_context.get('user_name', 'there')}! " if user_context.get('user_name') else ""
        context_prompt = f"{user_greeting}"
        
        if user_context.get('personalization_note'):
            context_prompt += f"User context: {user_context['personalization_note']} "
        
        if personality_context:
            context_prompt += f"{personality_context} "
        
        context_prompt += "Please provide a helpful, accurate, and personalized response based on your general knowledge while maintaining the specified personality traits and communication style."
        
        await send_update("ai_generation", 90, "Generating personalized greeting response...")
        
        openai_response = await generate_openai_answer(question, context_prompt)
        
        await send_update("ai_generation", 100, "Greeting response ready!")
        
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        # Update system statistics
        system_stats["questions_processed"] += 1
        system_stats["total_response_time"] += processing_time
        system_stats["cache_misses"] += 1
        
        # Use higher confidence for general queries
        general_confidence = {
            "confidence": 0.85,  # 85% confidence for general queries
            "confidence_factors": {
                "general_knowledge": 0.85,
                "response_quality": 0.8,
                "personality_alignment": 0.9
            },
            "confidence_level": "high",
            "reasoning": "General query response with high confidence"
        }
        
        response_data = {
            "answer": openai_response["answer"],
            "confidence": general_confidence["confidence"],
            "confidence_factors": general_confidence["confidence_factors"],
            "confidence_level": general_confidence["confidence_level"],
            "sources_used": [],
            "citations": [],
            "reasoning": f"General query response. {general_confidence['reasoning']}",
            "processing_time": round(processing_time / 1000, 3),
            "processing_time_ms": int(processing_time),
            "api_used": "openai",
            "vector_search_enabled": False,
            "embedding_model": OPENAI_EMBEDDING_MODEL,
            "results_count": len(all_docs),
            "tokens_used": openai_response["tokens_used"]
        }
        
        # Store chat history in Milvus
        try:
            if user_context.get('user_id'):
                await milvus_user_manager.create_chat_entry(
                    user_id=user_context['user_id'],
                    question=question,
                    answer=openai_response["answer"],
                    confidence=general_confidence["confidence"],
                    sources_used=[],
                    reasoning=response_data["reasoning"],
                    processing_time=processing_time
                )
        except Exception as e:
            logger.warning(f"Failed to store chat history in Milvus: {e}")
        
        return response_data
    
    await send_update("vector_search", 40, "Generating question embedding for semantic search...")
    
    # Generate question embedding for document search
    logger.info(f"Generating embedding for question: '{question}'")
    question_embedding = await generate_openai_embedding(question, f"question_{hashlib.md5(question.encode()).hexdigest()}")
    logger.info(f"Generated embedding with {len(question_embedding)} dimensions")
    
    await send_update("vector_search", 50, f"Performing semantic search across {len(all_docs)} documents...")
    
    # Perform vectorized document search using embeddings with more lenient threshold
    logger.info(f"Starting vectorized search with {len(all_docs)} documents")
    vectorized_results = await perform_vectorized_search(question, question_embedding, all_docs, relevance_threshold=0.15)
    
    logger.info(f"Vectorized search results: {len(vectorized_results['relevant_docs'])} relevant docs found, max relevance: {vectorized_results['max_relevance']:.3f}")
    
    await send_update("vector_search", 60, f"Found {len(vectorized_results['relevant_docs'])} relevant documents...")
    
    if not vectorized_results["relevant_docs"]:
        await send_update("ai_generation", 70, "No relevant documents found. Using web content and general AI knowledge...")
        
        # No relevant documents found - provide direct AI response with personality and web content
        user_greeting = f"Hello {user_context.get('user_name', 'there')}! " if user_context.get('user_name') else ""
        context_prompt = f"{user_greeting}"
        
        if user_context.get('personalization_note'):
            context_prompt += f"User context: {user_context['personalization_note']} "
        
        if personality_context:
            context_prompt += f"{personality_context} "
        
        # Add web content if available
        if web_search_used and web_content:
            context_prompt += f"\n\nWeb search results:\n{web_content}\n\n"
            context_prompt += "Please provide a comprehensive response based on the web search results above and your general knowledge. Reference specific information from the web sources when relevant."
        else:
            context_prompt += "Please provide a helpful and personalized response based on your general knowledge while maintaining the specified personality traits and communication style."
        
        await send_update("ai_generation", 90, "Generating comprehensive response...")
        
        openai_response = await generate_openai_answer(question, context_prompt)
        
        await send_update("ai_generation", 100, "Response generated successfully!")
        
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        # Update system statistics
        system_stats["questions_processed"] += 1
        system_stats["total_response_time"] += processing_time
        system_stats["cache_misses"] += 1
        
        # Use higher confidence for general knowledge responses
        general_confidence = {
            "confidence": 0.85,  # 85% confidence for general queries
            "confidence_factors": {
                "general_knowledge": 0.85,
                "response_quality": 0.8,
                "personality_alignment": 0.9
            },
            "confidence_level": "high",
            "reasoning": "General knowledge response (no relevant documents found) with high confidence"
        }
        
        # Include web sources in response if available
        sources_used = []
        citations = []
        reasoning_suffix = ""
        
        if web_search_used and web_sources:
            sources_used = [f"{source['title']} ({source['url']})" for source in web_sources]
            citations = [
                {
                    "source": source["url"],
                    "title": source["title"],
                    "content": f"Web search result: {source.get('snippet', 'No snippet available')}",
                    "confidence": 0.9,
                    "text": f"Web content from {source['title']}"
                }
                for source in web_sources
            ]
            reasoning_suffix = f" Used web search agent to gather information from {len(web_sources)} web source(s)."
        
        response_data = {
            "answer": openai_response["answer"],
            "confidence": general_confidence["confidence"],
            "confidence_factors": general_confidence["confidence_factors"],
            "confidence_level": general_confidence["confidence_level"],
            "sources_used": sources_used,
            "citations": citations,
            "reasoning": f"General knowledge response (no relevant documents found).{reasoning_suffix} {general_confidence['reasoning']}",
            "processing_time": round(processing_time / 1000, 3),
            "processing_time_ms": int(processing_time),
            "api_used": "openai",
            "vector_search_enabled": True,
            "embedding_model": OPENAI_EMBEDDING_MODEL,
            "results_count": len(all_docs),
            "tokens_used": openai_response["tokens_used"],
            "question_relevance": vectorized_results["max_relevance"],
            "web_search_enabled": web_search_enabled,
            "web_search_used": web_search_used,
            "web_sources": web_sources if web_search_used else []
        }
        
        # Store chat history in Milvus
        try:
            if user_context.get('user_id'):
                await milvus_user_manager.create_chat_entry(
                    user_id=user_context['user_id'],
                    question=question,
                    answer=openai_response["answer"],
                    confidence=general_confidence["confidence"],
                    sources_used=[],
                    reasoning=response_data["reasoning"],
                    processing_time=processing_time
                )
        except Exception as e:
            logger.warning(f"Failed to store chat history in Milvus: {e}")
        
        return response_data
    
    # Question is relevant - use vectorized results for enhanced context
    relevant_document_data = vectorized_results["relevant_docs"]
    
    await send_update("context_creation", 65, "Ranking and analyzing relevant documents...")
    
    # Step 1: Advanced document ranking using Synthesis Agent
    ranked_docs = await synthesis_agent.rank_results(question, relevant_document_data)
    
    await send_update("context_creation", 70, "Creating optimized context from relevant sources...")
    
    # Step 2: Create vectorized context from relevant documents
    sources_used = []
    
    # Add web sources first if available
    if web_search_used and web_sources:
        for source in web_sources:
            sources_used.append({
                "id": source["url"],
                "title": source["title"],
                "relevance_score": 0.95,  # High relevance for web search results
                "semantic_similarity": 0.95,
                "keyword_score": 0.95,
                "rank": len(sources_used) + 1,
                "web_search": True,
                "url": source["url"]
            })
    
    # Process vectorized results for sources tracking
    for i, doc_data in enumerate(relevant_document_data[:5]):  # Top 5 documents
        doc = doc_data["doc"]
        doc_fields = get_document_fields(doc, i)
        
        sources_used.append({
            "id": doc_fields['id'],
            "title": doc_fields['title'],
            "relevance_score": round(doc_data["relevance"], 3),
            "semantic_similarity": round(doc_data["semantic_similarity"], 3),
            "keyword_score": round(doc_data["keyword_score"], 3),
            "rank": len(sources_used) + 1,
            "vectorized_search": True
        })
    
    # Create optimized vectorized context
    context_text = await create_vectorized_context(question, relevant_document_data, max_context_length=3000)
    
    await send_update("ai_generation", 75, "Assembling context and preparing AI prompt...")
    
    # Step 3: Generate AI-powered answer with vectorized context and web content
    
    user_greeting = f"Hello {user_context.get('user_name', 'there')}! " if user_context.get('user_name') else ""
    user_context_text = f"User context: {user_context['personalization_note']} " if user_context.get('personalization_note') else ""
    
    # Combine document context with web content
    combined_context = context_text
    if web_search_used and web_content:
        combined_context += f"\n\nAdditional web search results:\n{web_content}"
    
    # Simplify prompt for GPT-5 compatibility
    enhanced_prompt = f"""Question: {question}

Context from documents and web sources:
{combined_context}

Please provide a comprehensive answer based on the provided documents and web content. Reference specific information from both the documents and scraped websites in your response."""
    
    # Add user context if available
    if user_context.get('user_name'):
        enhanced_prompt = f"Hello {user_context['user_name']}! " + enhanced_prompt
    
    # Add personality context if available
    if personality_context:
        enhanced_prompt += f"\n\nPlease respond with the following personality traits: {personality_context}"
    
    logger.info(f"Prompt length: {len(enhanced_prompt)} characters")
    logger.info(f"Context text length: {len(context_text)} characters")
    
    await send_update("ai_generation", 85, "Generating AI response with document context...")
    
    openai_response = await generate_openai_answer(question, enhanced_prompt)
    
    await send_update("ai_generation", 95, "Finalizing response and citations...")
    
    # Step 4: Calculate dynamic confidence using Synthesis Agent
    confidence_data = await synthesis_agent.calculate_dynamic_confidence(
        question, ranked_docs, openai_response["answer"]
    )
    
    # Step 5: Create enhanced citations
    relevant_ranked_docs = []
    for source in sources_used:
        for doc_data in ranked_docs:
            doc_fields = get_document_fields(doc_data["doc"])
            if doc_fields['id'] == str(source["id"]):
                relevant_ranked_docs.append(doc_data)
                break
    
    enhanced_citations = await synthesis_agent.create_enhanced_citations(
        relevant_ranked_docs, openai_response["answer"]
    )
    
    # Step 6: Perform quality check
    quality_check = await synthesis_agent.perform_quality_check(
        question, openai_response["answer"], enhanced_citations, confidence_data
    )
    
    await send_update("ai_generation", 100, "Response completed successfully!")
    
    # Calculate final processing time
    processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
    
    web_sources_count = len(web_sources) if web_search_used else 0
    doc_sources_count = len([s for s in sources_used if not s.get('web_search', False)])
    
    logger.info(f"Vectorized Search - Max Relevance: {vectorized_results['max_relevance']:.3f}, Documents Used: {doc_sources_count}, Web Sources: {web_sources_count}")
    logger.info(f"Synthesis Agent Analysis - Confidence: {confidence_data['confidence']:.3f} ({confidence_data['confidence_level']}), Quality: {quality_check['quality_level']}")
    
    reasoning_text = f"Vectorized search found {doc_sources_count} highly relevant documents (max relevance: {vectorized_results['max_relevance']:.3f})"
    if web_sources_count > 0:
        reasoning_text += f" and found {web_sources_count} relevant web source(s)"
    reasoning_text += f". {confidence_data['reasoning']}"
    
    response_data = {
        "answer": openai_response["answer"],
        "confidence": confidence_data["confidence"],
        "confidence_factors": confidence_data["confidence_factors"],
        "confidence_level": confidence_data["confidence_level"],
        "sources_used": sources_used,
        "citations": enhanced_citations,
        "reasoning": reasoning_text,
        "processing_time": round(processing_time / 1000, 3),
        "processing_time_ms": int(processing_time),
        "api_used": "openai",
        "vector_search_enabled": True,
        "embedding_model": OPENAI_EMBEDDING_MODEL,
        "results_count": len(all_docs),
        "tokens_used": openai_response["tokens_used"],
        "question_relevance": vectorized_results["max_relevance"],
        "web_search_enabled": web_search_enabled,
        "web_search_used": web_search_used,
        "web_sources": web_sources if web_search_used else [],
        "vectorized_analysis": {
            "search_method": vectorized_results["search_method"],
            "threshold_used": vectorized_results.get("threshold_used", 0.25),
            "documents_analyzed": vectorized_results["total_checked"],
            "relevant_documents_found": len(vectorized_results["relevant_docs"]),
            "avg_semantic_similarity": round(vectorized_results["avg_semantic_similarity"], 3),
            "top_similarities": [
                {
                    "title": get_document_fields(item["doc"])["title"] or "Untitled",
                    "relevance_score": round(item["relevance"], 3),
                    "semantic_similarity": round(item["semantic_similarity"], 3),
                    "keyword_score": round(item["keyword_score"], 3)
                }
                for item in vectorized_results["relevant_docs"][:3]
            ]
        },
        "synthesis_analysis": {
            "quality_check": quality_check,
            "top_document_scores": [
                {
                    "title": get_document_fields(doc_data["doc"])["title"] or "Untitled",
                    "relevance_score": round(doc_data["relevance"], 3)
                }
                for doc_data in relevant_document_data[:3]
            ]
        }
    }
    
    # Store chat history in Milvus
    try:
        if user_context.get('user_id'):
            await milvus_user_manager.create_chat_entry(
                user_id=user_context['user_id'],
                question=question,
                answer=openai_response["answer"],
                confidence=confidence_data["confidence"],
                sources_used=[source["title"] for source in sources_used],
                reasoning=response_data["reasoning"],
                processing_time=processing_time
            )
    except Exception as e:
        logger.warning(f"Failed to store chat history in Milvus: {e}")
    
    return response_data
