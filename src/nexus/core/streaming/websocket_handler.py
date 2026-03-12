"""
WebSocket handler for streaming responses.
"""

import logging
import asyncio
from typing import Optional, Dict, Any, Callable
from flask_socketio import SocketIO, emit, disconnect
from flask import request

logger = logging.getLogger(__name__)


class WebSocketHandler:
    """
    Handles WebSocket connections for streaming responses.
    
    Features:
    - Real-time token-by-token streaming
    - Connection management
    - Authentication via WebSocket
    - Progress updates
    - Error handling
    """
    
    def __init__(self, socketio: SocketIO, auth_manager=None):
        """
        Initialize WebSocket handler.
        
        Args:
            socketio: Flask-SocketIO instance
            auth_manager: Optional authentication manager
        """
        self.socketio = socketio
        self.auth_manager = auth_manager
        self.active_connections: Dict[str, Dict[str, Any]] = {}
        
        # Register event handlers
        self._register_handlers()
        
        logger.info("WebSocketHandler initialized")
    
    def _register_handlers(self):
        """Register WebSocket event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            client_id = request.sid
            logger.info(f"Client connected: {client_id}")
            
            self.active_connections[client_id] = {
                "connected_at": asyncio.get_event_loop().time(),
                "authenticated": False,
            }
            
            emit('connection_established', {
                'client_id': client_id,
                'message': 'Connected to TheNexus streaming'
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            client_id = request.sid
            logger.info(f"Client disconnected: {client_id}")
            
            if client_id in self.active_connections:
                del self.active_connections[client_id]
        
        @self.socketio.on('authenticate')
        def handle_authenticate(data):
            """Handle authentication."""
            client_id = request.sid
            api_key = data.get('api_key')
            
            if not api_key:
                emit('error', {'message': 'API key required'})
                disconnect()
                return
            
            # Validate API key if auth manager provided
            if self.auth_manager:
                api_key_obj = self.auth_manager.validate_key(api_key)
                if not api_key_obj:
                    emit('error', {'message': 'Invalid API key'})
                    disconnect()
                    return
                
                self.active_connections[client_id]['authenticated'] = True
                self.active_connections[client_id]['api_key'] = api_key_obj
                
                logger.info(f"Client authenticated: {client_id}")
                emit('authenticated', {'status': 'success'})
            else:
                # No auth required
                self.active_connections[client_id]['authenticated'] = True
                emit('authenticated', {'status': 'success'})
        
        @self.socketio.on('stream_request')
        def handle_stream_request(data):
            """Handle streaming inference request."""
            client_id = request.sid
            
            # Check authentication
            if client_id not in self.active_connections:
                emit('error', {'message': 'Not connected'})
                return
            
            if not self.active_connections[client_id].get('authenticated'):
                emit('error', {'message': 'Not authenticated'})
                return
            
            prompt = data.get('prompt')
            if not prompt:
                emit('error', {'message': 'Prompt required'})
                return
            
            logger.info(f"Stream request from {client_id}: {prompt[:50]}...")
            
            # Emit acknowledgment
            emit('stream_started', {'prompt': prompt})
    
    def stream_token(self, client_id: str, token: str, metadata: Optional[Dict] = None):
        """
        Stream a single token to client.
        
        Args:
            client_id: Client session ID
            token: Token to stream
            metadata: Optional metadata
        """
        if client_id in self.active_connections:
            self.socketio.emit('token', {
                'token': token,
                'metadata': metadata or {}
            }, room=client_id)
    
    def stream_chunk(self, client_id: str, chunk: str, metadata: Optional[Dict] = None):
        """
        Stream a text chunk to client.
        
        Args:
            client_id: Client session ID
            chunk: Text chunk to stream
            metadata: Optional metadata
        """
        if client_id in self.active_connections:
            self.socketio.emit('chunk', {
                'chunk': chunk,
                'metadata': metadata or {}
            }, room=client_id)
    
    def stream_progress(self, client_id: str, progress: float, message: str):
        """
        Send progress update to client.
        
        Args:
            client_id: Client session ID
            progress: Progress percentage (0-100)
            message: Progress message
        """
        if client_id in self.active_connections:
            self.socketio.emit('progress', {
                'progress': progress,
                'message': message
            }, room=client_id)
    
    def stream_complete(self, client_id: str, result: Dict[str, Any]):
        """
        Signal streaming completion.
        
        Args:
            client_id: Client session ID
            result: Final result data
        """
        if client_id in self.active_connections:
            self.socketio.emit('stream_complete', result, room=client_id)
            logger.info(f"Stream completed for client: {client_id}")
    
    def stream_error(self, client_id: str, error: str):
        """
        Send error to client.
        
        Args:
            client_id: Client session ID
            error: Error message
        """
        if client_id in self.active_connections:
            self.socketio.emit('error', {'message': error}, room=client_id)
            logger.error(f"Stream error for client {client_id}: {error}")
    
    def broadcast(self, event: str, data: Dict[str, Any]):
        """
        Broadcast to all connected clients.
        
        Args:
            event: Event name
            data: Event data
        """
        self.socketio.emit(event, data, broadcast=True)
        logger.debug(f"Broadcast event: {event}")
    
    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)
    
    def get_authenticated_count(self) -> int:
        """Get number of authenticated connections."""
        return sum(
            1 for conn in self.active_connections.values()
            if conn.get('authenticated')
        )
