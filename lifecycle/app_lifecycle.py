"""
Application lifecycle management for the agentic RAG system.

Copyright (c) 2025 Mike Tallent & Claude Sonnet - Geniusai.biz
All rights reserved. This software is proprietary and confidential.
"""

import logging
from contextlib import asynccontextmanager
from core.database import db_manager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    """Manage application lifespan events."""
    # Startup
    try:
        await db_manager.initialize()
        logger.info("Database manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database manager: {e}")
    
    yield
    
    # Shutdown
    try:
        await db_manager.close()
        logger.info("Database manager closed successfully")
    except Exception as e:
        logger.error(f"Error closing database manager: {e}")
