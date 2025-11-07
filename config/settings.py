"""Application configuration settings for Milvus-only architecture."""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    model_config = ConfigDict(extra='allow', env_file='.env')
    """Application settings loaded from environment variables."""
    
    # Milvus Configuration (Primary Database)
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    
    # AI Services
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    
    # Query Settings
    confidence_threshold: float = 0.7
    max_results: int = 10
    query_timeout: int = 30
    
    # Application
    secret_key: str = "dev-secret-key"
    debug: bool = True
    log_level: str = "INFO"
    
    # Cache Settings (in-memory)
    cache_ttl: int = 3600  # 1 hour default TTL
    max_cache_size: int = 1000  # Maximum cache entries
    
    # Performance Settings
    max_concurrent_requests: int = 100
    request_timeout: int = 30
    
    # Vector Search Settings
    vector_search_threshold: float = 0.7
    max_vector_results: int = 20
    
    # Content Processing
    max_content_length: int = 65535
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    
    # Session Management
    session_timeout: int = 3600  # 1 hour


settings = Settings()
