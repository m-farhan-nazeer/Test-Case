"""
API endpoint handlers for the agentic RAG system.

Copyright (c) 2025 Mike Tallent & Claude Sonnet - Geniusai.biz
All rights reserved. This software is proprietary and confidential.
"""

import logging
import json
import uuid
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from fastapi import HTTPException, Depends, File, UploadFile, Form
from fastapi.security import HTTPAuthorizationCredentials

from models.pydantic_models import (
    QuestionRequest, UserDetails, UserResponse, UserSignup, UserLogin, 
    AuthResponse, WebScrapingRequest, BulkWebScrapingRequest
)
from core.database import milvus_store, milvus_user_manager, milvus_system_manager, db_manager
from database.database_functions import get_db_connection
from database.milvus_functions import ensure_milvus_connection
from ai.openai_integration import generate_openai_embedding
from web.scraping_helpers import scrape_website_content, bulk_scrape_website
from files.file_processing import extract_pdf_text, extract_docx_text, extract_json_content, split_large_document
from processing.core_processing import process_question_unified
from utils.utility_functions import get_document_fields, parse_document_metadata
from search.relevance_checker import check_question_document_relevance
from content.extraction_functions import extract_most_relevant_excerpt, chunk_document_content, calculate_chunk_similarities

logger = logging.getLogger(__name__)

# Global variables - these will be set by main.py
in_memory_documents = []
document_counter = 1
system_stats = {}

def set_global_variables(memory_docs, doc_counter, stats):
    """Set global variables from main.py"""
    global in_memory_documents, document_counter, system_stats
    in_memory_documents = memory_docs
    document_counter = doc_counter
    system_stats = stats



async def ask_question_handler(request: QuestionRequest):
    """Ask a question endpoint with Lantern vector similarity search."""
    start_time = time.time()
    processing_time = 0.0  # Initialize at function start to ensure it's always available
    
    try:
        # Track API call
        system_stats["api_calls"] += 1
        
        # Process the question and get the response
        response = await process_question_unified(request.question, request.context)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Ensure the response has all required fields
        if not isinstance(response, dict):
            logger.error(f"Invalid response type from process_question_unified: {type(response)}")
            raise HTTPException(status_code=500, detail="Invalid response format")
        
        # Always set processing_time in response
        response["processing_time"] = processing_time
        
        # Add any missing fields with defaults
        if "confidence" not in response:
            response["confidence"] = 0.0
            
        if "sources_used" not in response:
            response["sources_used"] = []
            
        if "citations" not in response:
            response["citations"] = []
            
        if "reasoning" not in response:
            response["reasoning"] = "No reasoning provided"
        
        # Update system stats using the local processing_time variable
        system_stats["questions_processed"] += 1
        system_stats["total_response_time"] += processing_time
        
        return response
        
    except HTTPException as http_exc:
        # Ensure processing_time is calculated for HTTP exceptions
        if processing_time == 0.0:
            processing_time = (time.time() - start_time) * 1000
        logger.error(f"HTTP error processing question (took {processing_time:.2f}ms): {http_exc.detail}")
        raise http_exc
        
    except Exception as e:
        # Ensure processing_time is calculated for general exceptions
        if processing_time == 0.0:
            processing_time = (time.time() - start_time) * 1000
        logger.error(f"Error processing question (took {processing_time:.2f}ms): {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


async def get_stats_handler():
    """Get comprehensive system statistics from Milvus."""
    try:
        # Track API call
        system_stats["api_calls"] += 1
        
        # Calculate uptime
        uptime_seconds = (datetime.now(timezone.utc) - system_stats["start_time"]).total_seconds()
        uptime_hours = uptime_seconds / 3600
        
        # Calculate average response time in seconds
        avg_response_time_ms = (
            system_stats["total_response_time"] / system_stats["questions_processed"]
            if system_stats["questions_processed"] > 0 else 0.0
        )
        avg_response_time = avg_response_time_ms / 1000.0  # Convert to seconds
        
        # Calculate cache hit rate from in-memory cache
        total_cache_requests = system_stats["cache_hits"] + system_stats["cache_misses"]
        cache_hit_rate = (
            system_stats["cache_hits"] / total_cache_requests
            if total_cache_requests > 0 else 0.0
        )
        
        # Get Milvus info
        milvus_available = await ensure_milvus_connection()
        documents_count = 0
        total_size = "0 MB"
        
        if milvus_available:
            try:
                all_docs = await milvus_store.list_all_documents()
                documents_count = len(all_docs)
                
                # Estimate size based on content length
                estimated_size_bytes = sum(
                    doc.get("file_size", len(doc.get("content", "").encode('utf-8')))
                    for doc in all_docs
                )
                
                if estimated_size_bytes < 1024 * 1024:
                    total_size = f"{estimated_size_bytes / 1024:.1f} KB"
                else:
                    total_size = f"{estimated_size_bytes / (1024 * 1024):.1f} MB"
                    
            except Exception as e:
                logger.error(f"Milvus query error in stats: {e}")
                milvus_available = False
        
        # Use in-memory documents if Milvus unavailable
        if not milvus_available:
            documents_count = len(in_memory_documents)
            # Estimate size based on content length
            estimated_size_kb = sum(
                len(doc.get("content", "")) + len(doc.get("title", "")) 
                for doc in in_memory_documents
            ) / 1024
            total_size = f"{estimated_size_kb:.1f} KB (estimated)"
        
        # Determine system health
        system_health = "healthy"
        if not milvus_available:
            if len(in_memory_documents) == 0:
                system_health = "ready"  # System is ready but needs documents
            else:
                system_health = "limited"  # Working but without Milvus
        elif avg_response_time > 5000:  # > 5 seconds
            system_health = "slow"
        
        # If we have documents (either in Milvus or memory), system is healthy
        if documents_count > 0:
            system_health = "healthy"
        
        # Generate status message based on system state
        status_message = ""
        if system_health == "ready":
            status_message = "System is ready - upload documents to begin answering questions"
        elif system_health == "limited":
            status_message = f"Running with {documents_count} documents in memory (Milvus unavailable)"
        elif system_health == "healthy":
            status_message = f"System operational with {documents_count} documents in Milvus"
        elif system_health == "slow":
            status_message = "System running but response times are elevated"
        
        # Store current stats in Milvus for persistence
        if milvus_available:
            try:
                await milvus_system_manager.store_system_stat(
                    "current_stats", 
                    {
                        "questions_processed": system_stats["questions_processed"],
                        "documents_uploaded": system_stats["documents_uploaded"],
                        "api_calls": system_stats["api_calls"],
                        "uptime_hours": round(uptime_hours, 2)
                    },
                    "snapshot"
                )
            except Exception as e:
                logger.warning(f"Failed to store stats in Milvus: {e}")
        
        stats = {
            "total_questions_processed": system_stats["questions_processed"],
            "average_response_time": round(avg_response_time, 3),
            "average_response_time_ms": round(avg_response_time_ms, 2),
            "cache_hit_rate": round(cache_hit_rate * 100, 1),  # Convert to percentage
            "cache_status": "in_memory",
            "active_connections": 0,  # This would need to be passed in
            "system_health": system_health,
            "status_message": status_message,
            "documents_count": documents_count,
            "total_size": total_size,
            "queries_today": system_stats["questions_processed"],  # Simplified for now
            "milvus_available": milvus_available,
            "uptime_hours": round(uptime_hours, 2),
            "similarity_searches_performed": system_stats["similarity_searches"],
            "documents_uploaded": system_stats["documents_uploaded"],
            "websocket_connections_total": system_stats["websocket_connections"],
            "api_calls_total": system_stats["api_calls"],
            "memory_documents": len(in_memory_documents),
            "system_start_time": system_stats["start_time"].isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "detailed_stats": {
                "cache_hits": system_stats["cache_hits"],
                "cache_misses": system_stats["cache_misses"],
                "total_response_time_ms": round(system_stats["total_response_time"], 2),
                "average_response_time_seconds": round(avg_response_time, 3),
                "cache_size": len(db_manager.cache)
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


async def get_documents_handler():
    """Get all documents from the knowledge base."""
    try:
        # Try to get documents from Milvus first
        try:
            milvus_documents = await milvus_store.list_all_documents()
            if milvus_documents:
                return {
                    "documents": [
                        {
                            "id": doc["id"],
                            "title": doc["title"],
                            "content": doc["content"],
                            "summary": doc["summary"],
                            "metadata": parse_document_metadata(doc["metadata"]),
                            "created_at": doc["created_at"],
                            "size": doc.get("file_size", len(doc.get("content", ""))),
                            "status": "processed"
                        }
                        for doc in milvus_documents
                    ],
                    "total": len(milvus_documents),
                    "message": "Documents retrieved from Milvus"
                }
        except Exception as e:
            logger.warning(f"Could not retrieve documents from Milvus: {str(e)}")
        
        # Fallback to database
        conn = get_db_connection()
        if not conn:
            logger.info("Database not available, returning in-memory documents")
            return {
                "documents": [
                    {
                        "id": doc["id"],
                        "title": doc["title"],
                        "content": doc["content"],
                        "summary": doc["summary"],
                        "metadata": parse_document_metadata(doc["metadata"]),
                        "created_at": doc["created_at"].isoformat() if hasattr(doc["created_at"], 'isoformat') else str(doc["created_at"]),
                        "size": doc.get("size", len(doc.get("content", ""))),
                        "status": "processed"
                    }
                    for doc in in_memory_documents
                ],
                "total": len(in_memory_documents),
                "message": "Documents stored in memory (database not available)"
            }
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, title, content, summary, metadata, created_at
            FROM documents
            ORDER BY created_at DESC
        """)
        
        documents = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return {
            "documents": [
                {
                    "id": get_document_fields(doc)['id'],
                    "title": get_document_fields(doc)['title'],
                    "content": get_document_fields(doc)['content'],
                    "summary": get_document_fields(doc)['summary'],
                    "metadata": parse_document_metadata(get_document_fields(doc)['metadata']),
                    "created_at": get_document_fields(doc)['created_at'].isoformat() if hasattr(get_document_fields(doc)['created_at'], 'isoformat') else str(get_document_fields(doc)['created_at']),
                    "size": len(get_document_fields(doc)['content']),
                    "status": "processed"
                }
                for doc in documents
            ],
            "total": len(documents),
            "message": "Documents retrieved from database"
        }
        
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting documents: {str(e)}")


async def delete_document_handler(document_id: str):
    """Delete a document from the knowledge base."""
    try:
        # Convert document_id to appropriate type
        try:
            doc_id_int = int(document_id)
        except ValueError:
            doc_id_int = None
        
        deleted_from_milvus = False
        deleted_from_db = False
        deleted_from_memory = False
        
        # Try to delete from Milvus first
        if await ensure_milvus_connection():
            try:
                # Use the milvus_store delete method instead of direct Collection access
                success = await milvus_store.delete_document(document_id)
                if success:
                    deleted_from_milvus = True
                    logger.info(f"Document {document_id} deleted from Milvus")
                else:
                    logger.warning(f"Document {document_id} not found in Milvus")
                
            except Exception as e:
                logger.warning(f"Failed to delete from Milvus: {e}")
        
        # Try to delete from database
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id_int or document_id,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                cursor.close()
                conn.close()
                
                if deleted_count > 0:
                    deleted_from_db = True
                    logger.info(f"Document {document_id} deleted from database")
                
            except Exception as e:
                logger.error(f"Database deletion error: {e}")
                if conn:
                    conn.close()
        
        # Try to delete from in-memory store
        global in_memory_documents
        
        original_count = len(in_memory_documents)
        in_memory_documents = [
            doc for doc in in_memory_documents 
            if str(doc["id"]) != document_id and doc["id"] != doc_id_int
        ]
        
        if len(in_memory_documents) < original_count:
            deleted_from_memory = True
            logger.info(f"Document {document_id} deleted from memory")
        
        # Determine response based on what was deleted
        if deleted_from_milvus or deleted_from_db or deleted_from_memory:
            deleted_from = []
            if deleted_from_milvus:
                deleted_from.append("milvus")
            if deleted_from_db:
                deleted_from.append("database")
            if deleted_from_memory:
                deleted_from.append("memory")
            
            return {
                "status": "success",
                "message": f"Document {document_id} deleted successfully",
                "deleted_from": deleted_from,
                "document_id": document_id
            }
        else:
            return {
                "status": "not_found",
                "message": f"Document {document_id} not found",
                "document_id": document_id
            }
        
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


async def store_scraped_document(title: str, content: str, summary: str, metadata: Dict, embedding: List[float]) -> bool:
    """Store a scraped document in Milvus or memory fallback, splitting if too large."""
    try:
        # Check if document needs to be split
        max_content_length = 60000  # Leave buffer below 65535 limit
        
        if len(content) > max_content_length:
            logger.info(f"Document '{title}' is too large ({len(content)} chars), splitting into parts")
            
            # Split the document
            document_parts = await split_large_document(title, content, summary, metadata, max_content_length)
            
            # Store each part
            all_parts_stored = True
            for i, part in enumerate(document_parts):
                # Generate embedding for each part
                part_text = f"{part['title']} {part['content']} {part['summary']}"
                part_embedding = await generate_openai_embedding(part_text)
                
                if not isinstance(part_embedding, list) or len(part_embedding) == 0:
                    logger.error(f"Failed to generate embedding for part {i + 1}")
                    all_parts_stored = False
                    continue
                
                # Store the part
                part_stored = await store_single_document(
                    part['title'], 
                    part['content'], 
                    part['summary'], 
                    part['metadata'], 
                    part_embedding
                )
                
                if not part_stored:
                    all_parts_stored = False
            
            return all_parts_stored
        else:
            # Store as single document
            return await store_single_document(title, content, summary, metadata, embedding)
        
    except Exception as e:
        logger.error(f"Error storing document: {str(e)}")
        return False


async def store_single_document(title: str, content: str, summary: str, metadata: Dict, embedding: List[float]) -> bool:
    """Store a single document in Milvus or memory fallback."""
    try:
        # Ensure Milvus connection
        if await ensure_milvus_connection():
            try:
                doc_id = f"doc_{system_stats['documents_uploaded'] + 1}"
                created_at = datetime.now(timezone.utc).isoformat()
                
                await milvus_store.insert_document(
                    doc_id=doc_id,
                    title=title,
                    content=content,
                    summary=summary,
                    metadata=metadata,
                    embedding=embedding,
                    created_at=created_at
                )
                
                system_stats["documents_uploaded"] += 1
                
                # Store system stats
                await milvus_system_manager.store_system_stat(
                    "documents_uploaded", 
                    system_stats["documents_uploaded"], 
                    "counter"
                )
                
                logger.info(f"Document stored in Milvus: {title}")
                return True
                
            except Exception as milvus_error:
                logger.warning(f"Failed to store in Milvus: {milvus_error}")
        
        # Fallback to in-memory storage
        global document_counter, in_memory_documents
        doc_id = document_counter
        document_counter += 1
        
        in_memory_documents.append({
            "id": doc_id,
            "title": title,
            "content": content,
            "summary": summary,
            "metadata": metadata,
            "embedding": embedding,
            "created_at": datetime.now(timezone.utc)
        })
        
        system_stats["documents_uploaded"] += 1
        logger.info(f"Document stored in memory: {title}")
        return True
        
    except Exception as e:
        logger.error(f"Error storing single document: {str(e)}")
        return False
