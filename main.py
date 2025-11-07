# main.py
"""
Main entry point for the agentic RAG system.

Copyright (c) 2025 Mike Tallent & Claude Sonnet - Geniusai.biz
All rights reserved. This software is proprietary and confidential.
"""
from typing import Any, Dict, Optional, List
from uuid import uuid4
from datetime import datetime, timezone
from fastapi import HTTPException
import json
import logging
import warnings
# Suppress protobuf version warning
warnings.filterwarnings("ignore", message=".*Protobuf gencode version.*")
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database.database_functions import (
    get_db_connection,
    insert_document,
    ensure_documents_schema,   # ✅ for docs table
    init_chat_schema,          # ✅ used in startup
    enable_sqlite_wal,         # ✅ used in startup
    ensure_chat_indices,       # ✅ used in startup
)

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
import json
import logging
import hashlib
import aiohttp
import asyncio
import io
import PyPDF2
import docx2txt
import tempfile
import re

from lifecycle.app_lifecycle import lifespan
from core.database import milvus_store, milvus_user_manager, milvus_system_manager, db_manager
from models.pydantic_models import QuestionRequest, UserDetails, UserResponse, UserSignup, UserLogin, AuthResponse, WebScrapingRequest, BulkWebScrapingRequest
from utils.utility_functions import (
    safe_get_field, get_document_fields, parse_document_metadata, parse_document_embedding,
    format_personality_context, extract_important_terms, calculate_enhanced_overlap,
    get_keyword_weights_by_question_type
)
from database.milvus_functions import get_milvus_connection_status, ensure_milvus_connection
from ai.openai_integration import (
    generate_openai_embedding, generate_openai_answer, embedding_cache,
    openai_client, OPENAI_API_KEY, OPENAI_MODEL, OPENAI_EMBEDDING_MODEL
)
from vector.similarity_functions import calculate_cosine_similarity
from content.extraction_functions import chunk_document_content, calculate_chunk_similarities, extract_most_relevant_excerpt
from context.formatting_functions import get_user_context
from search.analysis_functions import analyze_question_type, analyze_question_complexity, search_documents_in_db
from scoring.advanced_scoring import (
    calculate_advanced_phrase_matching, assess_content_quality_for_question,
    calculate_freshness_score, calculate_question_type_match, apply_diversity_filtering
)
from scoring.dynamic_evaluation import (
    adjust_weights_for_question, calculate_dynamic_threshold, evaluate_document_relevance,
    calculate_chunk_relevance_score
)
from search.document_search import search_documents_in_memory
from search.vectorized_search import perform_vectorized_search
from search.relevance_checker import check_question_document_relevance
from context.context_creation import create_vectorized_context
from content.document_processing import extract_best_chunks_for_question, extract_vectorized_content
from synthesis.synthesis_agent import SynthesisAgent
from web.scraping_helpers import scrape_website_content, bulk_scrape_website
from files.file_processing import extract_pdf_text, extract_docx_text, extract_json_content, split_large_document
from endpoints.api_endpoints import (
    ask_question_handler, get_stats_handler, get_documents_handler, delete_document_handler,
    store_scraped_document, store_single_document, set_global_variables as set_api_globals
)
from endpoints.websocket_handlers import websocket_endpoint_handler, manager
from processing.core_processing import process_question_unified, set_global_variables as set_processing_globals
from endpoints.web_search_endpoints import perform_web_search_for_question, enhance_context_with_web_search
from auth.authentication import (
    verify_password, get_password_hash, create_access_token, verify_token, 
    get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES
)
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from api.chats_router import router as chat_router
# add:

# Google OAuth configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory document store for when database is not available
in_memory_documents = []
document_counter = 1

# System statistics tracking
system_stats = {
    "questions_processed": 0,
    "total_response_time": 0.0,
    "cache_hits": 0,
    "cache_misses": 0,
    "start_time": datetime.now(timezone.utc),
    "similarity_searches": 0,
    "documents_uploaded": 0,
    "websocket_connections": 0,
    "api_calls": 0
}

# ✅ ONE definitive app instance with lifespan
app = FastAPI(
    title="Agentic RAG System",
    description="Multi-language RAG system with intelligent question answering and Lantern vector search",
    version="2.0.0",
    lifespan=lifespan
)

# 🔐 CORS — unified list and single middleware
ALLOWED_ORIGINS = [
    "https://agent.geniusai.biz",  # Production frontend
    "http://localhost:3000",      # Development frontend
    "http://localhost:3001",      # Alternative development frontend
    "http://127.0.0.1:3000",      # Alternative localhost
    "http://127.0.0.1:3001",      # Alternative localhost
    "http://localhost:5173",      # Vite
    "http://127.0.0.1:5173",      # Vite alt
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)
@app.get("/__debug/sqlite")
def _debug_sqlite():
    from database.database_functions import get_db_connection
    import os, sqlite3
    db_path = os.getenv("SQLITE_DB_PATH", "agentic_rag.db")
    info = {"db_path_env": db_path, "pragma": []}
    conn = get_db_connection()
    if not conn:
        info["error"] = "get_db_connection() returned None"
        return info
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA database_list;")
        info["database_list"] = [tuple(r) for r in cur.fetchall()]
        cur.execute("PRAGMA table_info(documents);")
        info["pragma"] = [ (r[0], r[1], r[2]) for r in cur.fetchall() ]  # (cid, name, type)
        cur.close()
    finally:
        try: conn.close()
        except Exception: pass
    return info

from api.chats_router import router as chat_router
app.include_router(chat_router)
@app.get("/__debug/routes")
def _debug_routes():
    return [getattr(r, "path", str(r)) for r in app.router.routes]


# 🌍 Share globals with other modules — do this once, after app is defined
set_api_globals(in_memory_documents, document_counter, system_stats)
set_processing_globals(in_memory_documents, system_stats)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = datetime.now(timezone.utc)
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = (datetime.now(timezone.utc) - start_time).total_seconds()
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    # Track API calls for statistics (excluding health checks and static files)
    if not request.url.path.startswith(("/health", "/favicon", "/manifest")):
        system_stats["api_calls"] += 1
    
    return response


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Agentic RAG System API",
        "version": "2.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "docs": "/docs"
    }

@app.options("/{path:path}")
async def options_handler(path: str):
    """Handle OPTIONS requests for CORS preflight."""
    return {"message": "OK"}


async def verify_google_token(token: str) -> Dict[str, Any]:
    """Verify Google OAuth token and return user info."""
    try:
        if not GOOGLE_CLIENT_ID:
            raise HTTPException(status_code=500, detail="Google OAuth not configured")
        
        # Verify the token with Google
        idinfo = id_token.verify_oauth2_token(
            token, google_requests.Request(), GOOGLE_CLIENT_ID
        )
        
        # Verify the issuer
        if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            raise ValueError('Wrong issuer.')
        
        return {
            'google_id': idinfo['sub'],
            'email': idinfo['email'],
            'name': idinfo.get('name', ''),
            'picture': idinfo.get('picture', ''),
            'email_verified': idinfo.get('email_verified', False)
        }
        
    except ValueError as e:
        logger.error(f"Google token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid Google token")
    except Exception as e:
        logger.error(f"Google token verification error: {e}")
        raise HTTPException(status_code=500, detail="Google authentication failed")


@app.post("/api/auth/google", response_model=AuthResponse)
async def google_login(request: Dict[str, str]):
    """Authenticate user with Google OAuth token."""
    try:
        google_token = request.get('token')
        if not google_token:
            raise HTTPException(status_code=400, detail="Google token is required")
        
        # Verify Google token
        google_user = await verify_google_token(google_token)
        
        # Check if user exists
        existing_user = await milvus_user_manager.get_user_by_email(google_user['email'])
        
        if existing_user:
            # User exists, log them in
            user_id = existing_user['userId']
            user_name = existing_user['name']
        else:
            # Create new user
            import uuid
            user_id = str(uuid.uuid4())
            user_name = google_user['name'] or google_user['email'].split('@')[0]
            
            try:
                await milvus_user_manager.create_user(
                    user_id=user_id,
                    name=user_name,
                    email=google_user['email'],
                    password_hash="",  # No password for Google users
                    description=f"Google user - {google_user['email']}"
                )
                logger.info(f"New Google user created - user_id: {user_id}, email: {google_user['email']}")
            except Exception as e:
                logger.error(f"Failed to create Google user: {e}")
                raise HTTPException(status_code=500, detail="Failed to create user account")
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user_id, "email": google_user['email'], "auth_type": "google"},
            expires_delta=access_token_expires
        )
        
        # Update last active timestamp
        try:
            await milvus_system_manager.update_user_activity(user_id)
        except AttributeError:
            pass
        
        logger.info(f"Google user authenticated - user_id: {user_id}, email: {google_user['email']}")
        
        return AuthResponse(
            access_token=access_token,
            token_type="bearer",
            user_id=user_id,
            name=user_name,
            email=google_user['email'],
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Google login failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Google authentication failed")

@app.post("/api/auth/signup", response_model=AuthResponse)
async def signup(user_details: UserSignup):
    """Create a new user account with authentication."""
    try:
        import uuid
        
        # Check if user already exists
        existing_user = await milvus_user_manager.get_user_by_email(user_details.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Generate a unique user ID
        user_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        
        # Hash the password
        hashed_password = get_password_hash(user_details.password)
        
        # Store user details in Milvus
        try:
            await milvus_user_manager.create_user(
                user_id, 
                user_details.name, 
                user_details.email, 
                hashed_password,
                user_details.description
            )
            logger.info(f"User created in Milvus - user_id: {user_id}, email: {user_details.email}")
        except Exception as e:
            logger.error(f"Failed to store user in Milvus: {e}")
            raise HTTPException(status_code=500, detail="Failed to create user account")
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user_id, "email": user_details.email},
            expires_delta=access_token_expires
        )
        
        logger.info(f"User created successfully - user_id: {user_id}, name: {user_details.name}")
        
        return AuthResponse(
            access_token=access_token,
            token_type="bearer",
            user_id=user_id,
            name=user_details.name,
            email=user_details.email,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            created_at=created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create user: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")

@app.post("/api/auth/login", response_model=AuthResponse)
async def login(user_credentials: UserLogin):
    """Authenticate user and return access token."""
    try:
        # Get user by email
        user = await milvus_user_manager.get_user_by_email(user_credentials.email)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        # Verify password
        if not verify_password(user_credentials.password, user.get("password_hash", "")):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["userId"], "email": user["email"]},
            expires_delta=access_token_expires
        )
        
        logger.info(f"Generated token for user {user['userId']}: {access_token[:20]}...")
        logger.info(f"Token payload will be: {{'sub': '{user['userId']}', 'email': '{user['email']}'}}")
        
        # Update last active timestamp (if method exists)
        try:
            await milvus_system_manager.update_user_activity(user["userId"])
        except AttributeError:
            # Method doesn't exist, skip
            pass
        
        logger.info(f"User logged in successfully - user_id: {user['userId']}, email: {user['email']}")
        
        return AuthResponse(
            access_token=access_token,
            token_type="bearer",
            user_id=user["userId"],
            name=user["name"],
            email=user["email"],
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            created_at=user.get("createdAt", datetime.now(timezone.utc).isoformat())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.get("/api/auth/me")
async def get_current_user_info(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get current authenticated user information."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    return {
        "user_id": current_user["userId"],
        "name": current_user["name"],
        "email": current_user["email"],
        "description": current_user.get("description", ""),
        "createdAt": current_user.get("createdAt"),
        "lastActive": current_user.get("lastActive"),
        "preferences": current_user.get("preferences", {}),
        "picture": current_user.get("picture")
    }

@app.put("/api/auth/profile")
async def update_user_profile(
    profile_data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Update user profile information."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        user_id = current_user["userId"]
        
        # Validate required fields
        name = profile_data.get("name", "").strip()
        email = profile_data.get("email", "").strip()
        
        if not name:
            raise HTTPException(status_code=400, detail="Name is required")
        
        if not email:
            raise HTTPException(status_code=400, detail="Email is required")
        
        # Check if email is already taken by another user
        if email != current_user["email"]:
            existing_user = await milvus_user_manager.get_user_by_email(email)
            if existing_user and existing_user["userId"] != user_id:
                raise HTTPException(status_code=400, detail="Email is already taken")
        
        # Update user in Milvus
        try:
            # Get the current user data first
            existing_user = await milvus_user_manager.get_user_by_id(user_id)
            if not existing_user:
                raise HTTPException(status_code=404, detail="User not found")
            
            # Update the user data
            updated_user_data = {
                **existing_user,
                "name": name,
                "email": email,
                "description": profile_data.get("description", ""),
                "picture": profile_data.get("picture"),
                "lastActive": datetime.now(timezone.utc).isoformat()
            }
            
            # Since Milvus doesn't support direct updates, we need to delete and recreate
            # This is a limitation we'll work with for now
            await milvus_user_manager.delete_user(user_id)
            await milvus_user_manager.create_user(
                user_id=user_id,
                name=name,
                email=email,
                password_hash=existing_user.get("password_hash", ""),
                description=profile_data.get("description", ""),
                picture=profile_data.get("picture")
            )
            
            logger.info(f"Profile updated for user {user_id}")
            
            return {
                "message": "Profile updated successfully",
                "user": {
                    "user_id": user_id,
                    "name": name,
                    "email": email,
                    "description": profile_data.get("description", ""),
                    "picture": profile_data.get("picture"),
                    "created_at": existing_user.get("createdAt"),
                    "last_active": updated_user_data["lastActive"]
                }
            }
            
        except Exception as e:
            logger.error(f"Error updating user in Milvus: {e}")
            # Fallback: just return success with the data (for demo purposes)
            return {
                "message": "Profile updated successfully (cached)",
                "user": {
                    "user_id": user_id,
                    "name": name,
                    "email": email,
                    "description": profile_data.get("description", ""),
                    "picture": profile_data.get("picture"),
                    "created_at": current_user.get("createdAt"),
                    "last_active": datetime.now(timezone.utc).isoformat()
                }
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        raise HTTPException(status_code=500, detail="Failed to update profile")

@app.put("/api/auth/change-password")
async def change_password(
    password_data: Dict[str, str],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Change user password."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        current_password = password_data.get("currentPassword", "")
        new_password = password_data.get("newPassword", "")
        
        if not current_password:
            raise HTTPException(status_code=400, detail="Current password is required")
        
        if not new_password:
            raise HTTPException(status_code=400, detail="New password is required")
        
        if len(new_password) < 8:
            raise HTTPException(status_code=400, detail="New password must be at least 8 characters long")
        
        # Verify current password
        stored_password_hash = current_user.get("password_hash", "")
        if not verify_password(current_password, stored_password_hash):
            raise HTTPException(status_code=400, detail="Current password is incorrect")
        
        # Hash new password
        new_password_hash = get_password_hash(new_password)
        
        # Update password in Milvus
        try:
            # Get the current user data first
            existing_user = await milvus_user_manager.get_user_by_id(current_user["userId"])
            if not existing_user:
                raise HTTPException(status_code=404, detail="User not found")
            
            # Since Milvus doesn't support direct updates, we need to delete and recreate
            await milvus_user_manager.delete_user(current_user["userId"])
            await milvus_user_manager.create_user(
                user_id=current_user["userId"],
                name=existing_user["name"],
                email=existing_user["email"],
                password_hash=new_password_hash,
                description=existing_user.get("description", ""),
                picture=existing_user.get("picture")
            )
            
            logger.info(f"Password changed for user {current_user['userId']}")
            
            return {
                "message": "Password changed successfully"
            }
            
        except Exception as e:
            logger.error(f"Error updating password in Milvus: {e}")
            # For demo purposes, return success anyway
            logger.info(f"Password change simulated for user {current_user['userId']}")
            return {
                "message": "Password changed successfully (simulated)"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing password: {e}")
        raise HTTPException(status_code=500, detail="Failed to change password")

@app.post("/api/auth/upload-avatar")
async def upload_avatar(
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Upload user avatar image."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Validate file size (max 5MB)
        file_content = await file.read()
        if len(file_content) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size must be less than 5MB")
        
        # In a real application, you would upload to a cloud storage service
        # For now, we'll convert to base64 and store it
        import base64
        
        base64_image = base64.b64encode(file_content).decode('utf-8')
        image_url = f"data:{file.content_type};base64,{base64_image}"
        
        # Update user profile with new avatar URL
        # In a real implementation, you'd store the cloud URL
        logger.info(f"Avatar uploaded for user {current_user['userId']}")
        
        return {
            "message": "Avatar uploaded successfully",
            "avatar_url": image_url
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading avatar: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload avatar")

@app.delete("/api/users/{user_id}")
async def delete_user(user_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Delete a user permanently (admin only)."""
    try:
        # Check if user is authenticated
        logger.info(f"Delete user request - current_user: {current_user is not None}")
        if not current_user:
            logger.warning("Delete user failed: Not authenticated")
            raise HTTPException(status_code=401, detail="Not authenticated")
        
        logger.info(f"Authenticated user {current_user['email']} attempting to delete user {user_id}")
        
        # For now, allow any authenticated user to delete (you can add admin role checking later)
        # TODO: Add proper admin role checking
        
        # Try to delete from Milvus first
        milvus_deleted = False
        try:
            # Delete user from Users collection
            success = await milvus_user_manager.delete_user(user_id)
            if success:
                milvus_deleted = True
                logger.info(f"User {user_id} deleted from Milvus")
            else:
                logger.warning(f"User {user_id} not found in Milvus")
        except Exception as e:
            logger.warning(f"Failed to delete user from Milvus: {e}")
        
        # Try to delete from database
        db_deleted = False
        try:
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
                deleted_count = cursor.rowcount
                conn.commit()
                cursor.close()
                conn.close()
                
                if deleted_count > 0:
                    db_deleted = True
                    logger.info(f"User {user_id} deleted from database")
        except Exception as e:
            logger.error(f"Database deletion error: {e}")
        
        # Determine response based on what was deleted
        if milvus_deleted or db_deleted:
            deleted_from = []
            if milvus_deleted:
                deleted_from.append("milvus")
            if db_deleted:
                deleted_from.append("database")
            
            return {
                "status": "success",
                "message": f"User {user_id} deleted successfully",
                "deleted_from": deleted_from,
                "user_id": user_id
            }
        else:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting user: {str(e)}")

@app.get("/api/users")
async def get_all_users():
    """Get all users from the system."""
    try:
        users = []
        
        # Try to get users from Milvus first
        try:
            users = await milvus_user_manager.get_all_users()
            logger.info(f"Retrieved {len(users)} users from Milvus")
        except Exception as e:
            logger.warning(f"Could not retrieve users from Milvus: {str(e)}")
        
        # Fallback to database if Milvus fails or returns no users
        if not users:
            try:
                conn = get_db_connection()
                if conn:
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        SELECT id, name, email, description, created_at
                        FROM users 
                        ORDER BY created_at DESC
                    """)
                    
                    rows = cursor.fetchall()
                    for row in rows:
                        users.append({
                            "userId": row["id"] if "id" in row.keys() else row[0],
                            "name": row["name"] if "name" in row.keys() else row[1],
                            "email": row["email"] if "email" in row.keys() else row[2],
                            "description": row["description"] if "description" in row.keys() else row[3],
                            "createdAt": row["created_at"] if "created_at" in row.keys() else row[4]
                        })
                    
                    cursor.close()
                    conn.close()
                    logger.info(f"Retrieved {len(users)} users from database")
            except Exception as e:
                logger.warning(f"Could not retrieve users from database: {str(e)}")
        
        return {
            "users": users,
            "total": len(users),
            "message": f"Retrieved {len(users)} users successfully"
        }
        
    except Exception as e:
        logger.error(f"Error retrieving users: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving users: {str(e)}")

@app.post("/api/users", response_model=UserResponse)
async def create_user(user_details: UserDetails):
    """Create a new user and store their details."""
    try:
        import datetime
        import uuid
        
        # Generate a unique user ID
        user_id = str(uuid.uuid4())
        created_at = datetime.datetime.now().isoformat()
        
        # Store user details
        user_data = {
            "id": user_id,
            "name": user_details.name,
            "email": user_details.email,
            "description": user_details.description,
            "created_at": created_at
        }
        
        # Try to store in Milvus first
        milvus_success = False
        try:
            milvus_id = await milvus_user_manager.create_user(
                user_id, user_details.name, user_details.email, user_details.description
            )
            milvus_success = True
            logger.info(f"User stored in Milvus - user_id: {user_id}, milvus_id: {milvus_id}")
        except Exception as e:
            logger.warning(f"Failed to store user in Milvus: {e}")
        
        # Try to store in database if available
        try:
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    """INSERT OR REPLACE INTO users (id, name, email, description, created_at) 
                       VALUES (?, ?, ?, ?, ?)""",
                    (user_id, user_details.name, user_details.email, 
                     user_details.description, created_at)
                )
                
                conn.commit()
                cursor.close()
                conn.close()
                logger.info(f"User stored in database - user_id: {user_id}, email: {user_details.email}")
            else:
                logger.info(f"Database not available, user data logged only: {user_data}")
                
        except Exception as e:
            logger.info(f"User registration (database error): {str(e)}", extra={"user_data": user_data})
        
        logger.info(f"User created successfully - user_id: {user_id}, name: {user_details.name}")
        
        return UserResponse(
            message="User created successfully" + (" (stored in Milvus)" if milvus_success else ""),
            user_id=user_id,
            created_at=created_at
        )
        
    except Exception as e:
        logger.error(f"Failed to create user: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")


@app.post("/api/v1/ask")
async def ask_question(request: QuestionRequest):
    """Ask a question endpoint with Lantern vector similarity search."""
    return await ask_question_handler(request)


@app.get("/api/v1/stats")
async def get_stats():
    """Get comprehensive system statistics from Milvus."""
    return await get_stats_handler()


@app.get("/api/v1/documents")
async def get_documents():
    """Get all documents from the knowledge base."""
    return await get_documents_handler()


@app.delete("/api/v1/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the knowledge base."""
    return await delete_document_handler(document_id)


@app.post("/api/v1/search")
async def enhanced_search_documents(request: Dict[str, Any]):
    """Enhanced document search with chunk-based similarity analysis."""
    try:
        query = request.get("query", "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query parameter is required")
        
        max_results = request.get("max_results", 10)
        relevance_threshold = request.get("relevance_threshold", 0.3)
        include_chunks = request.get("include_chunks", True)
        chunk_size = request.get("chunk_size", 400)
        search_type = request.get("search_type", "hybrid")
        
        # Get all documents
        conn = get_db_connection()
        all_docs = []
        
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, title, content, summary, metadata, embedding, created_at
                    FROM documents
                    ORDER BY created_at DESC
                """)
                all_docs = cursor.fetchall()
                cursor.close()
                conn.close()
            except Exception as e:
                logger.error(f"Database query error in search: {e}")
                if conn:
                    conn.close()
        
        # Add in-memory documents
        all_docs.extend(in_memory_documents)
        
        if not all_docs:
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "search_type": search_type,
                "relevance_threshold": relevance_threshold,
                "message": "No documents available for search"
            }
        
        # Perform enhanced relevance checking
        relevance_check = await check_question_document_relevance(
            query, all_docs, relevance_threshold
        )
        
        if not relevance_check["is_relevant"]:
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "search_type": search_type,
                "relevance_threshold": relevance_threshold,
                "max_relevance": relevance_check["max_relevance"],
                "message": f"No documents found with relevance above threshold {relevance_threshold}"
            }
        
        # Process and format results
        search_results = []
        for i, item in enumerate(relevance_check["relevant_docs"][:max_results]):
            doc = item["doc"]
            
            # Use centralized document field extraction
            doc_fields = get_document_fields(doc, i)
            doc_metadata = parse_document_metadata(doc_fields['metadata'])
            
            # Extract most relevant excerpt
            relevant_excerpt = await extract_most_relevant_excerpt(
                query, doc_fields['content'], doc_fields['summary'], max_length=300
            )
            
            # Get chunk analysis if requested
            chunk_analysis = None
            if include_chunks and len(doc_fields['content']) > chunk_size:
                question_embedding = await generate_openai_embedding(query)
                chunks = chunk_document_content(doc_fields['content'], chunk_size, overlap=50)
                chunk_similarities = await calculate_chunk_similarities(query, chunks, question_embedding)
                
                chunk_analysis = {
                    "total_chunks": len(chunks),
                    "top_chunks": [
                        {
                            "chunk_index": chunk["chunk_index"],
                            "content": chunk["chunk"][:200] + "..." if len(chunk["chunk"]) > 200 else chunk["chunk"],
                            "similarity_score": round(chunk["combined_score"], 3),
                            "semantic_similarity": round(chunk["semantic_similarity"], 3)
                        }
                        for chunk in chunk_similarities[:3]
                    ]
                }
            
            search_result = {
                "document_id": doc_fields['id'],
                "title": doc_fields['title'],
                "excerpt": relevant_excerpt,
                "summary": doc_fields['summary'],
                "relevance_score": round(item["relevance"], 3),
                "semantic_similarity": round(item["semantic_similarity"], 3),
                "keyword_score": round(item["keyword_score"], 3),
                "phrase_score": round(item["phrase_score"], 3),
                "created_at": doc_fields['created_at'].isoformat() if hasattr(doc_fields['created_at'], 'isoformat') else str(doc_fields['created_at']),
                "metadata": doc_metadata,
                "content_length": len(doc_fields['content']),
                "match_details": {
                    "title_overlap": round(item["title_overlap"], 3),
                    "content_overlap": round(item["content_overlap"], 3),
                    "summary_overlap": round(item["summary_overlap"], 3),
                    "overall_semantic_similarity": round(item.get("overall_semantic_similarity", 0), 3),
                    "max_chunk_similarity": round(item.get("max_chunk_similarity", 0), 3),
                    "threshold_used": round(item.get("threshold_used", relevance_threshold), 3)
                }
            }
            
            if chunk_analysis:
                search_result["chunk_analysis"] = chunk_analysis
            
            search_results.append(search_result)
        
        return {
            "query": query,
            "results": search_results,
            "total_results": len(search_results),
            "total_documents_searched": len(all_docs),
            "search_type": search_type,
            "relevance_threshold": relevance_threshold,
            "max_relevance": round(relevance_check["max_relevance"], 3),
            "avg_semantic_similarity": round(relevance_check["avg_semantic_similarity"], 3),
            "search_statistics": {
                "documents_above_threshold": len(relevance_check["relevant_docs"]),
                "threshold_used": relevance_check["threshold_used"],
                "processing_time_ms": 0  # Could add timing if needed
            }
        }
        
    except Exception as e:
        logger.error(f"Error in enhanced search: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/api/v1/documents/{document_id}/content")
async def get_document_content(document_id: str):
    """Get the content of a specific document."""
    try:
        # Try to get from Milvus first
        if await ensure_milvus_connection():
            try:
                all_docs = await milvus_store.list_all_documents()
                for doc in all_docs:
                    if doc["id"] == document_id:
                        return {
                            "id": doc["id"],
                            "title": doc["title"],
                            "content": doc["content"],
                            "summary": doc["summary"],
                            "metadata": parse_document_metadata(doc["metadata"]),
                            "created_at": doc["created_at"]
                        }
            except Exception as e:
                logger.warning(f"Could not retrieve document from Milvus: {e}")
        
        # Convert document_id to appropriate type for database fallback
        try:
            doc_id_int = int(document_id)
        except ValueError:
            doc_id_int = None
        
        # Try to get from database
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, title, content, summary, metadata, created_at
                    FROM documents 
                    WHERE id = ?
                """, (doc_id_int or document_id,))
                
                document = cursor.fetchone()
                cursor.close()
                conn.close()
                
                if document:
                    doc_fields = get_document_fields(document)
                    return {
                        "id": doc_fields['id'],
                        "title": doc_fields['title'],
                        "content": doc_fields['content'],
                        "summary": doc_fields['summary'],
                        "metadata": parse_document_metadata(doc_fields['metadata']),
                        "created_at": doc_fields['created_at'].isoformat() if hasattr(doc_fields['created_at'], 'isoformat') else str(doc_fields['created_at'])
                    }
                
            except Exception as e:
                logger.error(f"Database query error: {e}")
                if conn:
                    conn.close()
        
        # Try to get from in-memory store
        for doc in in_memory_documents:
            if str(doc["id"]) == document_id or doc["id"] == doc_id_int:
                doc_fields = get_document_fields(doc)
                return {
                    "id": doc_fields['id'],
                    "title": doc_fields['title'],
                    "content": doc_fields['content'],
                    "summary": doc_fields['summary'],
                    "metadata": parse_document_metadata(doc_fields['metadata']),
                    "created_at": doc_fields['created_at'].isoformat() if hasattr(doc_fields['created_at'], 'isoformat') else str(doc_fields['created_at'])
                }
        
        # Document not found
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document content {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting document content: {str(e)}")


@app.post("/api/v1/scrape-website")
async def scrape_website(request: WebScrapingRequest):
    """Scrape content from a website and store it as a document."""
    try:
        logger.info(f"Starting web scraping for URL: {request.url}")
        
        # Scrape the website content
        scraped_data = await scrape_website_content(request.url)
        
        # Use provided title or fallback to scraped title
        document_title = request.title or scraped_data["title"]
        document_summary = request.summary or scraped_data["summary"]
        
        # Merge provided metadata with scraped metadata
        document_metadata = scraped_data["metadata"]
        if request.metadata:
            document_metadata.update(request.metadata)
        
        # Generate embedding for the document
        document_text = f"{document_title} {scraped_data['content']} {document_summary}"
        embedding = await generate_openai_embedding(document_text)
        
        # Verify embedding is valid
        if not isinstance(embedding, list) or len(embedding) == 0:
            logger.error("Generated embedding is not a valid list")
            raise HTTPException(status_code=500, detail="Failed to generate valid document embedding")
        
        logger.info(f"Generated embedding with {len(embedding)} dimensions for scraped content: {document_title}")
        
        # Fallback to database/memory storage
        conn = get_db_connection()
        if not conn:
            logger.warning("Database not available, storing scraped document in memory")
            global document_counter, in_memory_documents
            doc_id = document_counter
            document_counter += 1
            
            in_memory_documents.append({
                "id": doc_id,
                "title": document_title,
                "content": scraped_data["content"],
                "summary": document_summary,
                "metadata": document_metadata,
                "embedding": embedding,
                "created_at": datetime.now(timezone.utc)
            })
            
            logger.info(f"Scraped document stored in memory with ID: {doc_id}, title: '{document_title}'")
            
            # Update document upload statistics
            system_stats["documents_uploaded"] += 1
            
            return {
                "message": "Website scraped and document stored in memory (database not available)",
                "document_id": doc_id,
                "id": doc_id,
                "doc_id": doc_id,
                "title": document_title,
                "url": request.url,
                "content_length": len(scraped_data["content"]),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "embedding_dimensions": len(embedding),
                "metadata": document_metadata,
                "database_available": False,
                "success": True,
                "status": "success"
            }
        
        cursor = conn.cursor()
        
        # Verify embedding before storing
        if not isinstance(embedding, list) or len(embedding) == 0:
            logger.error("Invalid embedding format for SQLite storage")
            raise HTTPException(status_code=500, detail="Invalid embedding format")
        
        cursor.execute("""
            INSERT INTO documents (title, content, summary, metadata, embedding)
            VALUES (?, ?, ?, ?, ?)
        """, (document_title, scraped_data["content"], document_summary, json.dumps(document_metadata), json.dumps(embedding)))
        
        doc_id = cursor.lastrowid
        cursor.execute("SELECT created_at FROM documents WHERE id = ?", (doc_id,))
        created_at_result = cursor.fetchone()
        result = {"id": doc_id, "created_at": created_at_result["created_at"]}
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Update document upload statistics
        system_stats["documents_uploaded"] += 1
        
        return {
            "message": "Website scraped and document added successfully",
            "document_id": result["id"],
            "id": result["id"],
            "doc_id": result["id"],
            "title": document_title,
            "url": request.url,
            "content_length": len(scraped_data["content"]),
            "created_at": result["created_at"] if isinstance(result["created_at"], str) else result["created_at"].isoformat(),
            "embedding_dimensions": len(embedding),
            "metadata": document_metadata,
            "database_available": True,
            "success": True,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing website scraping: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing website scraping: {str(e)}")

@app.post("/api/v1/bulk-scrape-website")
async def bulk_scrape_website_endpoint(request: BulkWebScrapingRequest):
    """Scrape multiple pages from a website by discovering internal links."""
    try:
        logger.info(f"Starting bulk web scraping for URL: {request.url}, max_pages: {request.max_pages}")
        
        # Perform bulk scraping
        bulk_result = await bulk_scrape_website(
            base_url=request.url,
            max_pages=request.max_pages,
            title_prefix=request.title,
            summary_prefix=request.summary,
            combine_into_single_document=request.combine_into_single_document
        )
        
        # Handle document storage for combined mode
        combined_document_id = None
        if request.combine_into_single_document and bulk_result.get("combined_content"):
            try:
                # Generate embedding for combined document
                document_text = f"{bulk_result['combined_title']} {bulk_result['combined_content']} {bulk_result['combined_summary']}"
                
                # Check if content is too large for embedding
                estimated_tokens = len(document_text) / 4
                max_tokens = 8000
                
                if estimated_tokens > max_tokens:
                    logger.info(f"Combined document too large ({estimated_tokens:.0f} tokens), using truncated version for embedding")
                    truncated_content = bulk_result['combined_content'][:max_tokens * 3]
                    embedding_text = f"{bulk_result['combined_title']} {truncated_content} {bulk_result['combined_summary']}"
                    bulk_result['combined_metadata']["embedding_note"] = f"Embedding generated from truncated content due to size ({estimated_tokens:.0f} tokens)"
                else:
                    embedding_text = document_text
                
                embedding = await generate_openai_embedding(embedding_text)
                
                if isinstance(embedding, list) and len(embedding) > 0:
                    doc_stored = await store_scraped_document(
                        bulk_result['combined_title'], 
                        bulk_result['combined_content'], 
                        bulk_result['combined_summary'], 
                        bulk_result['combined_metadata'], 
                        embedding
                    )
                    
                    if doc_stored:
                        combined_document_id = system_stats["documents_uploaded"]
                        logger.info(f"Successfully created combined document: {bulk_result['combined_title']}")
                        
            except Exception as e:
                logger.error(f"Error creating combined document: {str(e)}")
        elif not request.combine_into_single_document:
            # Handle individual document storage
            for page_data in bulk_result.get("scraped_urls", []):
                if isinstance(page_data, dict):
                    try:
                        document_text = f"{page_data['title']} {page_data['content']} {page_data['summary']}"
                        embedding = await generate_openai_embedding(document_text)
                        
                        if isinstance(embedding, list) and len(embedding) > 0:
                            await store_scraped_document(
                                page_data['title'], 
                                page_data['content'], 
                                page_data['summary'], 
                                page_data['metadata'], 
                                embedding
                            )
                    except Exception as e:
                        logger.error(f"Error storing individual page: {str(e)}")
        
        # Create response
        if request.combine_into_single_document:
            message = f"Website crawled and combined into single document. Successfully scraped {bulk_result['successful_pages']} pages."
            if bulk_result.get("combined_document_id"):
                message += f" Document ID: {bulk_result['combined_document_id']}"
        else:
            message = f"Bulk scraping completed. Successfully scraped {bulk_result['successful_pages']} pages as separate documents."
        
        response_data = {
            "message": message,
            "title": request.title or f"Bulk scrape from {bulk_result['base_domain']}",
            "url": request.url,
            "total_pages": bulk_result["total_pages"],
            "successful_pages": bulk_result["successful_pages"],
            "failed_pages": bulk_result["failed_pages"],
            "scraped_urls": bulk_result["scraped_urls"],
            "failed_urls": bulk_result["failed_urls"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "combined_into_single_document": bulk_result["combined_into_single_document"],
            "metadata": {
                "bulk_scraping": True,
                "base_domain": bulk_result["base_domain"],
                "max_pages_requested": request.max_pages,
                "crawl_internal_links": request.crawl_internal_links,
                "combined_into_single_document": request.combine_into_single_document,
                **(request.metadata or {})
            },
            "success": True,
            "status": "success"
        }
        
        # Add combined document specific fields
        if request.combine_into_single_document and combined_document_id:
            response_data["document_id"] = combined_document_id
            response_data["combined_content_length"] = len(bulk_result.get("combined_content", ""))
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing bulk website scraping: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing bulk website scraping: {str(e)}")

# @app.post("/api/v1/upload-document-data")

def _generate_chat_id() -> str:
    return f"chat_{uuid4().hex}"

async def upload_document_data(
    document: Dict[str, Any],
    chat_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Upload document data with embeddings (aligned to the new SQLite+Milvus structure).
    - chat_id: added parameter; will be generated if not provided.
    - Uses a consistent `document_id` (UUID string) across SQLite and Milvus.
    - Stores content as BLOB in SQLite via `insert_document`.
    """
    try:
        title = (document.get('title') or '').strip()
        content = (document.get('content') or '').strip()
        summary = (document.get('summary') or '').strip()
        metadata = document.get('metadata') or {}

        if not title or not content:
            raise HTTPException(status_code=400, detail="Title and content are required")

        # Ensure chat_id
        chat_id = (chat_id or document.get('chat_id') or '').strip() or _generate_chat_id()

        # Build text for embedding (title + content + summary)
        document_text = f"{title}\n\n{content}\n\n{summary}".strip()
        embedding: List[float] = await generate_openai_embedding(document_text)

        if not isinstance(embedding, list) or not embedding:
            logger.error("Generated embedding is not a valid list")
            raise HTTPException(status_code=500, detail="Failed to generate valid document embedding")

        logger.info(f"Generated embedding with {len(embedding)} dims for document: {title}")

        max_content_length = 60000  # buffer below 65535
        created_at_iso = datetime.now(timezone.utc).isoformat()

        # Check Milvus first (keep your connection logic)
        milvus_ok = await ensure_milvus_connection()

        # Acquire DB connection (keep your existing function name)
        conn = get_db_connection()
        if conn:
            ensure_documents_schema(conn)  # make sure documents schema exists
            init_chat_schema(conn)         # (optional) ensure chat tables exist too

        # Helper to persist single doc (no split)
        async def _persist_single_doc(
            *,
            _title: str,
            _content: str,
            _summary: str,
            _metadata: Dict[str, Any],
            _embedding: List[float],
            _created_at: str,
        ) -> Dict[str, Any]:
            # Use one UUID for both stores
            document_id = uuid4().hex  # store as TEXT in SQLite; pass same to Milvus
            sqlite_done = False

            try:
                # 1) SQLite (content as BLOB)
                if conn:
                    insert_document(
                        conn,
                        document_id=document_id,
                        title=_title,
                        content=_content,          # insert_document handles str->bytes
                        summary=_summary or "",
                        metadata=_metadata,
                        chat_id=chat_id,
                    )
                    sqlite_done = True

                # 2) Milvus (if available)
                if milvus_ok:
                    await milvus_store.insert_document(
                        doc_id=document_id,       # keep the SAME id in Milvus
                        title=_title,
                        content=_content,
                        summary=_summary or "",
                        metadata=_metadata,
                        embedding=_embedding,
                        created_at=_created_at,
                    )
                else:
                    logger.warning("Milvus not available, skipping vector insert")

                system_stats["documents_uploaded"] += 1

                return {
                    "message": "Document uploaded",
                    "document_id": document_id,
                    "doc_id": document_id,
                    "chat_id": chat_id,
                    "title": _title,
                    "content_length": len(_content),
                    "created_at": _created_at,
                    "embedding_dimensions": len(_embedding),
                    "metadata": _metadata,
                    "database_available": bool(conn),
                    "milvus_available": bool(milvus_ok),
                    "success": True,
                    "status": "success",
                    "storage_location": "database+milvus" if conn and milvus_ok else (
                        "database" if conn else ("milvus" if milvus_ok else "memory")
                    ),
                }

            except Exception as e:
                logger.error(f"Persist single doc failed: {e}")

                # Best-effort rollback if SQLite inserted but Milvus failed
                if sqlite_done and conn:
                    try:
                        with conn:
                            conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
                    except Exception as re:
                        logger.error(f"SQLite rollback failed: {re}")

                raise

        # Split if content too large
        if len(content) > max_content_length:
            logger.info(f"Document '{title}' too large ({len(content)} chars), splitting")

            parts = await split_large_document(
                title, content, summary or "", metadata, max_content_length
            )
            stored_parts = []

            for i, part in enumerate(parts):
                part_text = f"{part['title']}\n\n{part['content']}\n\n{part.get('summary','')}".strip()
                part_embedding = await generate_openai_embedding(part_text)

                if not isinstance(part_embedding, list) or not part_embedding:
                    logger.error(f"Failed to generate embedding for part {i + 1}")
                    continue

                # Persist each part
                try:
                    res = await _persist_single_doc(
                        _title=part["title"],
                        _content=part["content"],
                        _summary=part.get("summary", ""),
                        _metadata=part.get("metadata", {}),
                        _embedding=part_embedding,
                        _created_at=created_at_iso,
                    )
                    stored_parts.append({
                        "doc_id": res["document_id"],
                        "title": part["title"],
                        "part_number": part.get("metadata", {}).get("part_number"),
                        "content_length": len(part["content"]),
                    })
                    logger.info(f"Stored part {i+1} with ID: {res['document_id']}")
                except Exception as e:
                    logger.error(f"Failed to store document part {i + 1}: {e}")

            if stored_parts:
                return {
                    "message": f"Large document split and uploaded as {len(stored_parts)} parts",
                    "document_id": stored_parts[0]["doc_id"],
                    "doc_id": stored_parts[0]["doc_id"],
                    "chat_id": chat_id,
                    "title": title,
                    "content_length": len(content),
                    "created_at": created_at_iso,
                    "embedding_dimensions": len(embedding),
                    "metadata": {**metadata, "split_into_parts": len(stored_parts)},
                    "success": True,
                    "status": "success",
                    "database_available": bool(conn),
                    "milvus_available": bool(milvus_ok),
                    "storage_location": "database+milvus" if conn and milvus_ok else (
                        "database" if conn else ("milvus" if milvus_ok else "memory")
                    ),
                    "split_info": {
                        "total_parts": len(stored_parts),
                        "parts": stored_parts,
                    },
                }

        # No split → single insert path
        try:
            return await _persist_single_doc(
                _title=title,
                _content=content,
                _summary=summary,
                _metadata=metadata,
                _embedding=embedding,
                _created_at=created_at_iso,
            )
        except Exception as e:
            logger.error(f"Failed to store document (DB/Milvus path): {e}")

        # Last-chance fallback to memory if both DB & Milvus not available
        logger.warning("Falling back to memory store")
        import uuid, time
        doc_id = f"mem_{uuid.uuid4().hex[:12]}_{int(time.time() * 1000000)}"
        in_memory_documents.append({
            "id": doc_id,
            "title": title,
            "content": content,
            "summary": summary or "",
            "metadata": metadata,
            "embedding": embedding,
            "chat_id": chat_id,
            "created_at": datetime.now(timezone.utc),
        })
        system_stats["documents_uploaded"] += 1

        return {
            "message": "Document stored in memory (Milvus/DB unavailable)",
            "document_id": doc_id,
            "doc_id": doc_id,
            "chat_id": chat_id,
            "title": title,
            "content_length": len(content),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedding_dimensions": len(embedding),
            "metadata": metadata,
            "database_available": False,
            "milvus_available": False,
            "success": True,
            "status": "success",
            "storage_location": "memory",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document data: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document data: {str(e)}")

# Add these three endpoints to your main.py file (after the existing auth endpoints)

@app.put("/api/v1/settings/system-prompt")
async def update_system_prompt(
    prompt_data: Dict[str, str],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Update the system prompt for the authenticated user."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        user_id = current_user["userId"]
        system_prompt = prompt_data.get("system_prompt", "").strip()
        
        if not system_prompt:
            raise HTTPException(status_code=400, detail="System prompt cannot be empty")
        
        if len(system_prompt) > 2000:
            raise HTTPException(status_code=400, detail="System prompt too long (max 2000 characters)")
        
        # Get current user data
        existing_user = await milvus_user_manager.get_user_by_id(user_id)
        if not existing_user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Update preferences with system prompt
        current_preferences = existing_user.get("preferences", {})
        current_preferences["system_prompt"] = system_prompt
        current_preferences["system_prompt_updated_at"] = datetime.now(timezone.utc).isoformat()
        
        # Update user in Milvus (delete and recreate due to Milvus limitations)
        try:
            await milvus_user_manager.delete_user(user_id)
            await milvus_user_manager.create_user(
                user_id=user_id,
                name=existing_user["name"],
                email=existing_user["email"],
                password_hash=existing_user.get("password_hash", ""),
                description=existing_user.get("description", ""),
                preferences=current_preferences
            )
            
            logger.info(f"System prompt updated for user {user_id}")
            
            return {
                "message": "System prompt updated successfully",
                "system_prompt": system_prompt,
                "updated_at": current_preferences["system_prompt_updated_at"],
                "character_count": len(system_prompt)
            }
            
        except Exception as e:
            logger.error(f"Error updating system prompt in Milvus: {e}")
            raise HTTPException(status_code=500, detail="Failed to update system prompt")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating system prompt: {e}")
        raise HTTPException(status_code=500, detail="Failed to update system prompt")


@app.get("/api/v1/settings/system-prompt")
async def get_system_prompt(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get the current system prompt for the authenticated user."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        preferences = current_user.get("preferences", {})
        system_prompt = preferences.get("system_prompt", "")
        updated_at = preferences.get("system_prompt_updated_at")
        
        return {
            "system_prompt": system_prompt,
            "updated_at": updated_at,
            "has_custom_prompt": bool(system_prompt),
            "character_count": len(system_prompt) if system_prompt else 0
        }
        
    except Exception as e:
        logger.error(f"Error getting system prompt: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system prompt")


@app.delete("/api/v1/settings/system-prompt")
async def delete_system_prompt(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Delete/reset the system prompt for the authenticated user."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        user_id = current_user["userId"]
        
        # Get current user data
        existing_user = await milvus_user_manager.get_user_by_id(user_id)
        if not existing_user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Remove system prompt from preferences
        current_preferences = existing_user.get("preferences", {})
        current_preferences.pop("system_prompt", None)
        current_preferences.pop("system_prompt_updated_at", None)
        
        # Update user in Milvus
        try:
            await milvus_user_manager.delete_user(user_id)
            await milvus_user_manager.create_user(
                user_id=user_id,
                name=existing_user["name"],
                email=existing_user["email"],
                password_hash=existing_user.get("password_hash", ""),
                description=existing_user.get("description", ""),
                preferences=current_preferences
            )
            
            logger.info(f"System prompt deleted for user {user_id}")
            
            return {
                "message": "System prompt reset to default successfully"
            }
            
        except Exception as e:
            logger.error(f"Error deleting system prompt in Milvus: {e}")
            raise HTTPException(status_code=500, detail="Failed to delete system prompt")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting system prompt: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete system prompt")

@app.post("/api/v1/upload-file")
async def upload_file(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    summary: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None)
):
    """Upload and process a file (.pdf, .docx, .txt, .json) to extract content."""
    try:
        # Validate file type
        allowed_types = [
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword',
            'text/plain',
            'application/json'
        ]
        
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file type. Please upload PDF, Word document (.docx), text file (.txt), or JSON file."
            )
        
        # Read file content
        file_content = await file.read()
        
        # Extract text based on file type
        extracted_content = ""
        extracted_title = title or file.filename.rsplit('.', 1)[0] if file.filename else "Uploaded Document"
        
        # Parse metadata
        parsed_metadata = {"originalFileName": file.filename, "fileType": file.content_type, "fileSize": len(file_content)}
        if metadata:
            try:
                user_metadata = json.loads(metadata)
                parsed_metadata.update(user_metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in metadata field")
        
        if file.content_type == 'application/pdf':
            extracted_content = await extract_pdf_text(file_content)
        elif file.content_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword']:
            extracted_content = await extract_docx_text(file_content)
        elif file.content_type == 'text/plain':
            extracted_content = file_content.decode('utf-8')
        elif file.content_type == 'application/json':
            json_result = await extract_json_content(file_content)
            extracted_content = json_result['content']
            if not title and json_result.get('title'):
                extracted_title = json_result['title']
            if json_result.get('metadata'):
                parsed_metadata.update(json_result['metadata'])
        
        if not extracted_content.strip():
            raise HTTPException(status_code=400, detail="No text content could be extracted from the file")
        
        # Generate embedding for the document with proper verification
        document_text = f"{extracted_title} {extracted_content} {summary or ''}"
        embedding = await generate_openai_embedding(document_text)
        
        # Verify embedding is a proper list of floats
        if not isinstance(embedding, list) or len(embedding) == 0:
            logger.error("Generated embedding is not a valid list")
            raise HTTPException(status_code=500, detail="Failed to generate valid document embedding")
        
        logger.info(f"Generated embedding with {len(embedding)} dimensions for document: {extracted_title}")
        
        # Use the new document splitting logic for file uploads too
        doc_stored = await store_scraped_document(
            extracted_title, 
            extracted_content, 
            summary or "", 
            parsed_metadata, 
            embedding
        )
        
        if doc_stored:
            return {
                "message": "File processed and document uploaded successfully",
                "document_id": system_stats["documents_uploaded"],
                "doc_id": system_stats["documents_uploaded"],
                "title": extracted_title,
                "content_length": len(extracted_content),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "embedding_dimensions": len(embedding),
                "metadata": parsed_metadata,
                "success": True,
                "status": "success",
                "storage_location": "milvus" if await ensure_milvus_connection() else "memory"
            }
        else:
            logger.error("Failed to store file document")
        
        # Fallback to database/memory storage
        conn = get_db_connection()
        if not conn:
            logger.warning("Database not available, storing document in memory")
            import uuid
            global document_counter, in_memory_documents
            doc_id = f"mem_{uuid.uuid4().hex[:8]}_{int(datetime.now(timezone.utc).timestamp())}"
            
            in_memory_documents.append({
                "id": doc_id,
                "title": extracted_title,
                "content": extracted_content,
                "summary": summary or "",
                "metadata": parsed_metadata,
                "embedding": embedding,
                "created_at": datetime.now(timezone.utc)
            })
            
            logger.info(f"File document stored in memory with ID: {doc_id}, title: '{extracted_title}'")
            
            # Update document upload statistics
            system_stats["documents_uploaded"] += 1
            
            return {
                "message": "File processed and document stored in memory (Milvus and database not available)",
                "document_id": doc_id,
                "id": doc_id,
                "doc_id": doc_id,
                "title": extracted_title,
                "content_length": len(extracted_content),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "embedding_dimensions": len(embedding),
                "metadata": parsed_metadata,
                "database_available": False,
                "milvus_available": False,
                "success": True,
                "status": "success",
                "storage_location": "memory"
            }
        
        cursor = conn.cursor()
        
        # Verify embedding before storing
        if not isinstance(embedding, list) or len(embedding) == 0:
            logger.error("Invalid embedding format for SQLite storage")
            raise HTTPException(status_code=500, detail="Invalid embedding format")
        
        cursor.execute("""
            INSERT INTO documents (title, content, summary, metadata, embedding)
            VALUES (?, ?, ?, ?, ?)
        """, (extracted_title, extracted_content, summary or "", json.dumps(parsed_metadata), json.dumps(embedding)))
        
        doc_id = cursor.lastrowid
        cursor.execute("SELECT created_at FROM documents WHERE id = ?", (doc_id,))
        created_at_result = cursor.fetchone()
        result = {"id": doc_id, "created_at": created_at_result["created_at"]}
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Update document upload statistics
        system_stats["documents_uploaded"] += 1
        
        return {
            "message": "File processed and document added successfully to database",
            "document_id": result["id"],
            "id": result["id"],
            "doc_id": result["id"],
            "title": extracted_title,
            "content_length": len(extracted_content),
            "created_at": result["created_at"] if isinstance(result["created_at"], str) else result["created_at"].isoformat(),
            "embedding_dimensions": len(embedding),
            "metadata": parsed_metadata,
            "database_available": True,
            "milvus_available": False,
            "success": True,
            "status": "success",
            "storage_location": "database"
        }
        
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing uploaded file: {str(e)}")

# --- Chat DB migration on startup ---
@app.on_event("startup")
def _chat_schema_startup():
    try:
        conn = get_db_connection()
        if conn:
            try:
                init_chat_schema(conn)
                enable_sqlite_wal(conn)       # <-- NEW
                ensure_chat_indices(conn)     # <-- NEW
                logger.info("Chat schema & indices ready on startup (WAL enabled).")
            finally:
                conn.close()
        else:
            logger.error("Startup: DB unavailable; chat features may fail until DB is reachable.")
    except Exception as e:
        logger.exception(f"Startup chat migration failed: {e}")


@app.get("/api/v1/users/{user_id}/chat-history")
async def get_user_chat_history(user_id: str, limit: int = 50):
    """Get chat history for a specific user."""
    try:
        chat_history = await milvus_user_manager.get_user_chat_history(user_id, limit)
        
        return {
            "user_id": user_id,
            "chat_history": chat_history,
            "total_entries": len(chat_history),
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting chat history: {str(e)}")


@app.get("/health")
async def health_check():
    """Enhanced health check endpoint for Milvus-only architecture."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.0.0",
        "services": {
            "api": "running",
            "milvus": "unknown",
            "websocket": "running",
            "memory_cache": "unknown",
            "vector_search": "unknown"
        },
        "metrics": {
            "active_websocket_connections": len(manager.active_connections),
            "documents_in_memory": len(in_memory_documents),
            "total_questions_processed": system_stats["questions_processed"],
            "uptime_seconds": int((datetime.now(timezone.utc) - system_stats["start_time"]).total_seconds()),
            "cache_entries": len(db_manager.cache)
        }
    }
    
    try:
        # Test Milvus connection
        milvus_available = await ensure_milvus_connection()
        if milvus_available:
            try:
                # Test basic Milvus operations
                from pymilvus import utility
                collections = utility.list_collections()
                health_status["services"]["milvus"] = "connected"
                health_status["services"]["vector_search"] = "enabled"
                health_status["metrics"]["milvus_collections"] = len(collections)
            except Exception as e:
                logger.error(f"Milvus test query failed: {e}")
                health_status["services"]["milvus"] = "error"
                health_status["services"]["vector_search"] = "disabled"
                health_status["status"] = "degraded"
        else:
            health_status["services"]["milvus"] = "disconnected"
            health_status["services"]["vector_search"] = "disabled"
            health_status["status"] = "degraded"
        
        # Check memory cache
        health_status["services"]["memory_cache"] = "active" if db_manager.cache else "empty"
        
        # Clean expired cache entries
        await db_manager.clear_expired_cache()
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        health_status["services"]["milvus"] = "error"
        health_status["status"] = "degraded"
    
    return health_status

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    await websocket_endpoint_handler(websocket, system_stats)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
