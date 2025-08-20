import asyncio
from typing import Optional, Set

from fastapi import WebSocket

from .face_app_manager import FaceAppManager


class ConnectionManager:
    """WebSocket connection manager"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.face_app_manager: Optional[FaceAppManager] = None
        self.stream_task: Optional[asyncio.Task] = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

        # Start face detection service when first client connects
        # if len(self.active_connections) == 1:
        #     await self.start_face_detection()

    async def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

        # Stop face detection service when no clients connected
        if len(self.active_connections) == 0:
            print("All clients disconnected, stopping face detection service.")
            await self.stop_face_detection()

    async def send_frame(self, frame_data: str):
        """Send frame to all connected clients"""
        disconnected_clients = set()
        for connection in self.active_connections:
            try:
                await connection.send_json({"type": "frame", "data": frame_data})
            except:
                disconnected_clients.add(connection)

        # Remove disconnected clients
        self.active_connections -= disconnected_clients

    async def send_log(self, log_data: dict):
        """Send log to all connected clients"""
        disconnected_clients = set()
        for connection in self.active_connections:
            try:
                await connection.send_json({"type": "log", "data": log_data})
            except:
                disconnected_clients.add(connection)

        self.active_connections -= disconnected_clients

    async def broadcast_message(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected_clients = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error sending message to client: {e}")
                disconnected_clients.add(connection)

        self.active_connections -= disconnected_clients

    async def start_face_detection(self):
        """Start face detection service"""
        if self.face_app_manager is None:
            self.face_app_manager = FaceAppManager(self)
            self.stream_task = asyncio.create_task(self.face_app_manager.run())
            await self.broadcast_message({"type": "status", "message": "Face detection service started"})

    async def stop_face_detection(self):
        """Stop face detection service"""
        if self.face_app_manager:
            await self.face_app_manager.stop()
            if self.stream_task:
                self.stream_task.cancel()
                try:
                    await self.stream_task
                except asyncio.CancelledError:
                    pass
            self.face_app_manager = None
            self.stream_task = None
            await self.broadcast_message({"type": "status", "message": "Face detection service stopped"})
