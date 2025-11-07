"""
Pydantic models for request/response validation.

Copyright (c) 2025 Mike Tallent & Claude Sonnet - Geniusai.biz
All rights reserved. This software is proprietary and confidential.
"""

from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict, Any


class QuestionRequest(BaseModel):
    question: str
    context: Optional[Dict[str, Any]] = None
    web_search_enabled: Optional[bool] = False
    
    @field_validator('question')
    @classmethod
    def question_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()


class UserDetails(BaseModel):
    name: str
    email: str
    description: str = ""


class UserSignup(BaseModel):
    name: str
    email: str
    password: str
    description: Optional[str] = ""
    
    @field_validator('name')
    @classmethod
    def name_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()
    
    @field_validator('email')
    @classmethod
    def email_must_be_valid(cls, v):
        if not v or '@' not in v:
            raise ValueError('Valid email address is required')
        return v.strip().lower()
    
    @field_validator('password')
    @classmethod
    def password_must_be_strong(cls, v):
        if not v or len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v


class UserLogin(BaseModel):
    email: str
    password: str
    
    @field_validator('email')
    @classmethod
    def email_must_be_valid(cls, v):
        if not v or '@' not in v:
            raise ValueError('Valid email address is required')
        return v.strip().lower()


class UserResponse(BaseModel):
    message: str
    user_id: str
    created_at: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    name: str
    email: str
    expires_in: int
    created_at: Optional[str] = None


class WebScrapingRequest(BaseModel):
    url: str
    title: Optional[str] = None
    summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @field_validator('url')
    @classmethod
    def url_must_be_valid(cls, v):
        if not v or not v.strip():
            raise ValueError('URL cannot be empty')
        
        # Add http:// if no protocol is specified
        if not v.startswith(('http://', 'https://')):
            v = 'https://' + v
        
        # Basic URL validation
        from urllib.parse import urlparse
        parsed = urlparse(v)
        if not parsed.netloc:
            raise ValueError('Invalid URL format')
        
        return v.strip()


class BulkWebScrapingRequest(BaseModel):
    url: str
    title: Optional[str] = None
    summary: Optional[str] = None
    max_pages: int = 10
    crawl_internal_links: bool = True
    combine_into_single_document: bool = True
    metadata: Optional[Dict[str, Any]] = None
    
    @field_validator('url')
    @classmethod
    def url_must_be_valid(cls, v):
        if not v or not v.strip():
            raise ValueError('URL cannot be empty')
        
        # Add http:// if no protocol is specified
        if not v.startswith(('http://', 'https://')):
            v = 'https://' + v
        
        # Basic URL validation
        from urllib.parse import urlparse
        parsed = urlparse(v)
        if not parsed.netloc:
            raise ValueError('Invalid URL format')
        
        return v.strip()
    
    @field_validator('max_pages')
    @classmethod
    def max_pages_must_be_valid(cls, v):
        if v < 1 or v > 100:
            raise ValueError('max_pages must be between 1 and 100')
        return v
