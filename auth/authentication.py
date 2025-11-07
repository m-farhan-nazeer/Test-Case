"""
Authentication utilities for the agentic RAG system.

Copyright (c) 2025 Mike Tallent & Claude Sonnet - Geniusai.biz
All rights reserved. This software is proprietary and confidential.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from core.database import milvus_user_manager, db_manager

logger = logging.getLogger(__name__)

# Authentication configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 * 24 * 60  # 30 days

# Password hashing - use explicit bcrypt configuration for better compatibility
pwd_context = CryptContext(
    schemes=["bcrypt"], 
    deprecated="auto",
    bcrypt__rounds=12
)

# Security
security = HTTPBearer(auto_error=False)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode a JWT token."""
    try:
        logger.info(f"Verifying token with SECRET_KEY: {SECRET_KEY[:10]}... and algorithm: {ALGORITHM}")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        logger.info(f"Token decoded successfully: {payload}")
        return payload
    except jwt.ExpiredSignatureError as e:
        logger.warning(f"Token has expired: {e}")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected token verification error: {e}")
        return None


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[Dict[str, Any]]:
    """Get current user from JWT token."""
    if not credentials:
        logger.info("No credentials provided in Authorization header")
        return None
    
    logger.info(f"Received token: {credentials.credentials[:20]}...")
    
    # Verify token
    payload = verify_token(credentials.credentials)
    if not payload:
        logger.warning("Token verification failed - invalid or expired token")
        return None
    
    logger.info(f"Token payload: {payload}")
    
    user_id = payload.get("sub")
    if not user_id:
        logger.warning("No user ID found in token payload")
        return None
    
    logger.info(f"Looking up user by ID: {user_id}")
    
    # Get user from Milvus
    try:
        # Check if Milvus is connected
        if not db_manager.milvus_connected:
            logger.warning("Milvus not connected, cannot retrieve user")
            return None
            
        user = await milvus_user_manager.get_user_by_id(user_id)
        if user:
            logger.info(f"User found: {user['email']}")
            return user
        else:
            logger.warning(f"User {user_id} not found in Milvus")
            return None
    except Exception as e:
        logger.error(f"Error retrieving user {user_id} from Milvus: {e}")
        return None
