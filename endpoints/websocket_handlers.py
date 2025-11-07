"""
WebSocket handlers for real-time communication.

Copyright (c) 2025 Mike Tallent & Claude Sonnet - Geniusai.biz
All rights reserved. This software is proprietary and confidential.
"""

import json
import logging
from datetime import datetime, timezone
from typing import List
from fastapi import WebSocket, WebSocketDisconnect
from processing.core_processing import process_question_unified
from ai.openai_integration import embedding_cache

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            self.disconnect(websocket)

    async def send_processing_update(self, step: str, progress: int, message: str, websocket: WebSocket):
        """Send processing update to client."""
        try:
            update = {
                "type": "processing_update",
                "step": step,
                "progress": progress,
                "message": message,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            await websocket.send_text(json.dumps(update))
        except Exception as e:
            logger.error(f"Error sending processing update: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


async def websocket_endpoint_handler(websocket: WebSocket, system_stats: dict):
    """WebSocket endpoint for real-time communication."""
    try:
        await manager.connect(websocket)
        
        # Clear embedding cache for WebSocket requests too
        global embedding_cache
        embedding_cache.clear()
        
        # Track WebSocket connection
        system_stats["websocket_connections"] += 1
        
        # Send connection confirmation (with error handling)
        try:
            welcome_message = {
                "type": "connected",
                "message": "WebSocket connection established",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            await manager.send_personal_message(json.dumps(welcome_message), websocket)
        except Exception as e:
            logger.error(f"Failed to send welcome message: {e}")
        
        while True:
            data = await websocket.receive_text()
            try:
                # Parse incoming JSON message
                message = json.loads(data)
                message_type = message.get('type', 'unknown')
                
                if message_type == 'question':
                    # Handle real-time question processing
                    question = message.get('question', '')
                    if not question:
                        error_response = {
                            "type": "error",
                            "message": "Question is required",
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                        await manager.send_personal_message(json.dumps(error_response), websocket)
                        continue
                    
                    # Send initial processing status
                    await manager.send_processing_update(
                        "analyzing", 10, "Starting to analyze your question...", websocket
                    )
                    
                    # Use unified question processing with WebSocket context and updates
                    context = message.get('context', {})
                    if 'personality' in message:
                        context['personality'] = message['personality']
                    
                    # Process using unified function with real-time updates
                    # Import here to avoid circular imports
                    from processing.core_processing import process_question_unified_with_updates
                    
                    response_data = await process_question_unified_with_updates(
                        question, 
                        context,
                        websocket=websocket,
                        update_callback=manager.send_processing_update
                    )
                    
                    # Convert to WebSocket format
                    answer_response = {
                        "type": "answer",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        **response_data
                    }
                    
                    await manager.send_personal_message(json.dumps(answer_response), websocket)
                
                elif message_type == 'ping':
                    # Handle ping messages
                    pong_response = {
                        "type": "pong",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    await manager.send_personal_message(json.dumps(pong_response), websocket)
                
                else:
                    # Handle other message types
                    response = {
                        "type": "response",
                        "message": f"Received message type: {message_type}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "original": message
                    }
                    await manager.send_personal_message(json.dumps(response), websocket)
                
            except json.JSONDecodeError:
                # Handle non-JSON messages
                error_response = {
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                await manager.send_personal_message(json.dumps(error_response), websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)


