import asyncio
from typing import Optional, Set

from fastapi import WebSocket

from .face_app_manager import FaceAppManager
from .face_registration_manager import FaceRegistrationManager


class ConnectionManager:
    """WebSocket connection manager"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.face_app_manager: Optional[FaceAppManager] = None
        self.stream_task: Optional[asyncio.Task] = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"New client connected: {websocket.client}")

    async def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        print(f"Client disconnected: {websocket.client}")

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
                print(f"Sending message to client {connection.client}: {message}")
                await connection.send_json(message)
            except Exception as e:
                print(f"Error sending message to client: {e}")
                disconnected_clients.add(connection)

        self.active_connections -= disconnected_clients

    async def start_face_detection(self):
        """Start face detection service"""
        if self.face_app_manager is None:
            print("Starting face detection service...")
            self.face_app_manager = FaceAppManager(self)
            self.stream_task = asyncio.create_task(self.face_app_manager.run())
            await self.broadcast_message({"type": "status", "message": "Face detection service started"})

    async def stop_face_detection(self):
        """Stop face detection service"""
        if self.face_app_manager:
            print("Stopping face detection service...")
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

    async def start_video_stream(self):
        """Start video stream service"""
        if self.face_app_manager is None:
            print("Starting video stream service...")
            self.face_app_manager = FaceRegistrationManager(self)
            self.stream_task = asyncio.create_task(self.face_app_manager.run())
            await self.broadcast_message({"type": "status", "message": "Video stream service started"})

    async def stop_video_stream(self):
        """Stop video stream service"""
        if self.face_app_manager:
            print("Stopping video stream service...")
            await self.face_app_manager.stop()
            if self.stream_task:
                self.stream_task.cancel()
                try:
                    await self.stream_task
                except asyncio.CancelledError:
                    pass
            self.face_app_manager = None
            self.stream_task = None
            await self.broadcast_message({"type": "status", "message": "Video stream service stopped"})
