"""
Context formatting functions for AI responses.

Copyright (c) 2025 Mike Tallent & Claude Sonnet - Geniusai.biz
All rights reserved. This software is proprietary and confidential.
"""

import logging
from typing import Dict, Any
from database.database_functions import get_db_connection
from database.milvus_functions import ensure_milvus_connection
from core.database import milvus_user_manager
from utils.utility_functions import safe_get_field

logger = logging.getLogger(__name__)


async def get_user_context():
    """Get user context from Weaviate or database to include in AI queries."""
    try:
        # Try Milvus first
        try:
            users = await milvus_user_manager.get_all_users()
            if users:
                # Get the most recent user
                user = users[0]  # Assuming they're sorted by creation date
                return {
                    "user_id": user.get("userId"),
                    "user_name": user.get("name"),
                    "user_email": user.get("email"),
                    "user_description": user.get("description", ""),
                    "personalization_note": f"The user's name is {user.get('name')}. {user.get('description', '') if user.get('description') else 'No additional user context provided.'}"
                }
        except Exception as e:
            logger.warning(f"Could not retrieve user context from Milvus: {str(e)}")
        
        # Fallback to database
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT name, email, description FROM users ORDER BY created_at DESC LIMIT 1")
            
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if row:
                return {
                    "user_name": safe_get_field(row, "name"),
                    "user_email": safe_get_field(row, "email"),
                    "user_description": safe_get_field(row, "description"),
                    "personalization_note": f"The user's name is {safe_get_field(row, 'name')}. {safe_get_field(row, 'description') if safe_get_field(row, 'description') else 'No additional user context provided.'}"
                }
    except Exception as e:
        logger.warning(f"Could not retrieve user context: {str(e)}")
    
    return {}
