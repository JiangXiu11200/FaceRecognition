"""
Building a face recognition application as a microservice using FastAPI.
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Form, HTTPException, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocketDisconnect
from sqlalchemy.orm import Session

from app_server.connection_manager import ConnectionManager
from app_server.db.database import Base, engine, get_db
from app_server.db.models import FaceRecognitionConfig, SystemConfig, VideoConfig
from app_server.db.schemas import FaceRecognitionConfigBase, SystemConfigBase, VideoConfigBase
from app_server.utils.image_tools import base64_to_bgr
from app_server.utils.minio_client import MinioClient
from app_server.utils.preview_camera import capture_image_from_camera
from package.face_feature_extractor import FaceFeatureExtractor

load_dotenv()

UPLOAD_TO_S3 = os.getenv("UPLOAD_TO_S3", "False").lower() == "true"


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
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                await manager.disconnect(websocket)
                break
            if message.get("type") == "start_detection":
                print("⚠️ Starting face detection...")
                await manager.start_face_detection()
            elif message.get("type") == "stop_detection":
                print("⚠️ Stopping face detection...")
                await manager.stop_face_detection()
            elif message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif message.get("type") == "start_video_stream":
                print("⚠️ Starting video stream...")
                await manager.start_video_stream()
            elif message.get("type") == "stop_video_stream":
                print("⚠️ Stopping video stream...")
                await manager.stop_video_stream()
            else:
                await websocket.send_text(json.dumps({"type": "error", "message": "Unknown message type"}))

    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        print(f"Error in WebSocket connection: {e}")
        await manager.disconnect(websocket)


@app.get("/api/health")
async def health_check():
    """Check connection and service status."""
    return {
        "status": "ok",
        "face_detection_running": manager.face_app_manager is not None,
        "active_connections": len(manager.active_connections),
    }


@app.get("/api/face-reco-config")
async def read_face_reco_config(db: Session = Depends(get_db)):
    """Read face recognition configuration."""
    config = db.query(FaceRecognitionConfig).first()
    if config:
        return config
    else:
        raise HTTPException(status_code=404, detail="Face recognition configuration not found")


@app.post("/api/face-reco-config")
async def update_face_reco_config(face_reco_config: FaceRecognitionConfigBase, db: Session = Depends(get_db)):
    """Update or create face recognition configuration."""
    db_config = db.query(FaceRecognitionConfig).first()
    if db_config:
        for key, value in face_reco_config.model_dump().items():
            setattr(db_config, key, value)
    else:
        db_config = FaceRecognitionConfig(**face_reco_config.model_dump())
        db.add(db_config)
    db.commit()
    db.refresh(db_config)
    return db_config


@app.get("/api/debug")
async def debug_info(system_config: Session = Depends(get_db)):
    """Get face recognition service debug mode settings."""
    system_config = system_config.query(SystemConfig).first()
    if system_config:
        return system_config
    else:
        raise HTTPException(status_code=404, detail="System configuration not found")


@app.post("/api/debug")
async def update_debug_info(system_config: SystemConfigBase, db: Session = Depends(get_db)):
    """Update face recognition service debug mode settings."""
    db_config = db.query(SystemConfig).first()
    if db_config:
        for key, value in system_config.model_dump().items():
            setattr(db_config, key, value)
    else:
        db_config = SystemConfig(**system_config.model_dump())
        db.add(db_config)
    db.commit()
    db.refresh(db_config)
    return db_config


@app.get("/api/video-config")
async def read_video_config(db: Session = Depends(get_db)):
    """Read face recognition configuration."""
    config = db.query(VideoConfig).first()
    if config:
        return config
    else:
        raise HTTPException(status_code=404, detail="Face recognition configuration not found")


@app.post("/api/video-config")
async def update_video_config(video_config: VideoConfigBase, db: Session = Depends(get_db)):
    """Update or create face recognition configuration."""
    db_config = db.query(VideoConfig).first()
    if db_config:
        for key, value in video_config.model_dump().items():
            setattr(db_config, key, value)
    else:
        db_config = VideoConfig(**video_config.model_dump())
        db.add(db_config)
    db.commit()
    db.refresh(db_config)
    return db_config


@app.post("/api/register-face")
async def register_face(base64_face_image: str = Form(...), name: str = Form(...), db: Session = Depends(get_db)):
    """
    Register a new face with the provided image and name.
    """
    config = db.query(FaceRecognitionConfig).first()
    if not config:
        raise HTTPException(status_code=404, detail="Face recognition configuration not found")
    try:
        face_feature_extractor = FaceFeatureExtractor(
            feature_csv_path=config.face_model,
            dlib_predictor_path=config.dlib_predictor_path,
            dlib_recognition_model_path=config.dlib_recognition_model_path,
            user_name=name,
        )
        image_bytes, face_image_nparry = base64_to_bgr(base64_face_image)
        convert_status, message = face_feature_extractor.get_face_roi(face_image_nparry)
        if not convert_status:
            raise HTTPException(status_code=503, detail=message.get("error"))
        face_roi = message.get("face_roi")
        feature_extraction_status, message = face_feature_extractor.feature_extraction(face_roi)
        if not feature_extraction_status:
            raise HTTPException(status_code=503, detail=message.get("error"))
        saved_status, message = face_feature_extractor.save_feature(message.get("face_descriptor"))
        if not saved_status:
            raise HTTPException(status_code=503, detail=message.get("error"))

        object_name = None
        if UPLOAD_TO_S3:
            object_name = f"{name}_register_face.jpg"
            try:
                upload_status, message = MinioClient.upload_object(
                    bucket_name="user-registration",
                    absolute_path_or_binary=image_bytes,
                    s3_object_key=object_name,
                    is_binary=True,
                )
            except Exception as e:
                print(f"Error uploading to MinIO S3: {e}")
                raise HTTPException(status_code=503, detail={"error": str(e)})
            if not upload_status:
                face_feature_extractor.delete_feature(config.face_model, name)
                raise HTTPException(status_code=503, detail=message.get("error"))

        return Response(
            status_code=201,
            content=json.dumps(
                {
                    "status": "success",
                    "s3_object_key": object_name,
                    "registr_name": name,
                    "message": f"Face registered for {name}",
                }
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})


@app.post("/api/delete-registered-face/{user_name}")
async def delete_registered_face(user_name: str, db: Session = Depends(get_db)):
    """
    Delete a registered face by name.
    """
    config = db.query(FaceRecognitionConfig).first()
    if not config:
        raise HTTPException(status_code=404, detail="Face recognition configuration not found")

    delete_status, message = FaceFeatureExtractor.delete_feature(config.face_model, user_name)

    MinioClient.delete_directory(bucket_name="user-registration", directory_name=user_name)

    if delete_status:
        return Response(
            status_code=200,
            content=json.dumps(message),
        )
    else:
        raise HTTPException(status_code=404, detail=message)


@app.get("/api/preview-camera/")
async def preview_camera(db: Session = Depends(get_db)):
    """Preview camera stream."""
    video_config = db.query(VideoConfig).first()
    if not video_config:
        raise HTTPException(status_code=404, detail="Video configuration not found")
    if not video_config.rtsp and video_config.web_camera is None:
        raise HTTPException(status_code=400, detail="No RTSP URL or webcam index configured")
    try:
        frame_bytes = await capture_image_from_camera(
            camera_source=video_config.web_camera if video_config.web_camera is not None else video_config.rtsp,
            resize=(video_config.image_width, video_config.image_height),
        )
        upload_status, message = MinioClient.upload_object(
            bucket_name="temporary-data",
            absolute_path_or_binary=frame_bytes,
            s3_object_key="/preview/preview_camera_image.jpg",
            is_binary=True,
        )
        if not upload_status:
            raise HTTPException(status_code=503, detail=message.get("error"))
        get_url_status, message = MinioClient.get_object_url(
            bucket_name="temporary-data",
            s3_object_key="/preview/preview_camera_image.jpg",
        )
        if not get_url_status:
            raise HTTPException(status_code=503, detail=message.get("error"))
        return {"preview_image_url": message.get("url")}

    except Exception as e:
        print(f"Error in preview_camera: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})
