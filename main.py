"""
Building a face recognition application as a microservice using FastAPI.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocketDisconnect

from app_server.connection_manager import ConnectionManager
from app_server.db.database import Base, engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager"""
    # Startup
    print("Starting FastAPI application...")

    # Create necessary directories
    Path("captured_faces").mkdir(exist_ok=True)
    Path("models/dlib").mkdir(parents=True, exist_ok=True)
    Path("models/face_recognition").mkdir(parents=True, exist_ok=True)

    Base.metadata.create_all(bind=engine)  # TODO: For development create tables, in feture use `Alembic` for migrations

    yield

    # Shutdown and stop the application
    print("Shutting down FastAPI application...")

    if manager.face_app_manager:
        await manager.face_app_manager.stop()


manager = ConnectionManager()

app = FastAPI(
    title="Face Recognition API", version="0.0.1", documentation_url="/docs", redoc_url="/redoc", lifespan=lifespan
)

# FIXME: For development only, restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message.get("type") == "start_detection":
                print("⚠️ Starting face detection...")
                await manager.start_face_detection()
            elif message.get("type") == "stop_detection":
                print("⚠️ Stopping face detection...")
                await manager.stop_face_detection()
            elif message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        print(f"Error in WebSocket connection: {e}")
        await manager.disconnect(websocket)
