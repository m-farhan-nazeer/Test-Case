"""
Milvus database functions.

Copyright (c) 2025 Mike Tallent & Claude Sonnet - Geniusai.biz
All rights reserved. This software is proprietary and confidential.
"""

import logging
from core.database import db_manager

logger = logging.getLogger(__name__)


async def get_milvus_connection_status():
    """Check Milvus connection status."""
    return db_manager.milvus_connected


async def ensure_milvus_connection():
    """Ensure Milvus connection is active."""
    try:
        if not db_manager.milvus_connected:
            logger.info("Attempting to initialize Milvus connection...")
            await db_manager.initialize()
            if db_manager.milvus_connected:
                logger.info("Milvus connection successfully established")
            else:
                logger.warning("Milvus connection failed to establish")
        return db_manager.milvus_connected
    except Exception as e:
        logger.error(f"Error ensuring Milvus connection: {e}")
        return False
