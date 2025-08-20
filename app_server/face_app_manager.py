import asyncio
import csv
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from queue import Queue

import dlib
import numpy as np

from face_detection import RunMode

from .database import SessionLocal
from .models import FaceRecognitionConfig, SystemConfig, SystemLogs, VideoConfig
from .utils import post_log_to_server


@dataclass
class VideoConfigData:
    __solts__ = [
        "rtsp",
        "web_camera",
        "image_height",
        "image_width",
        "detection_range_start_point",
        "detection_range_end_point",
    ]
    rtsp: str | None
    web_camera: int | None
    image_height: int
    image_width: int
    detection_range_start_point: list
    detection_range_end_point: list


@dataclass
class SystemConfigData:
    __slots__ = ["debug"]
    debug: bool


@dataclass
class RecoConfigData:
    enable: bool
    set_mode: bool
    enable_blink_detection: bool
    dlib_predictor: str
    dlib_recognition_model: str
    face_model: str
    minimum_bounding_box_height: int
    minimum_face_detection_score: float
    eyes_detection_brightness_threshold: int
    eyes_detection_brightness_value: list
    sensitivity: float
    consecutive_prediction_intervals: int
    registered_face_descriptor: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        # Initialize dlib models
        self.dlib_predictor = dlib.shape_predictor(self.dlib_predictor)
        self.dlib_recognition_model = dlib.face_recognition_model_v1(self.dlib_recognition_model)
        # Load face features
        self.load_face_features()

    def load_face_features(self):
        face_features = []
        if os.path.isfile(self.face_model):
            with open(self.face_model) as model:
                rows = csv.reader(model)
                for row in rows:
                    face_features.append(np.array(row, dtype=float))
        else:
            # Create the directory if it does not exist
            directory_path = os.path.dirname(self.face_model)
            os.makedirs(directory_path, exist_ok=True)
            with open(self.face_model, mode="a"):
                pass
        self.registered_face_descriptor = np.array(face_features)


class ConfigAdapter:
    """Adapter to convert database config to FaceApp config format"""

    def __init__(self):
        self.load_from_db()

    def load_from_db(self):
        """Load configuration from database"""
        db = SessionLocal()
        try:
            sys_config = db.query(SystemConfig).first()
            if sys_config:
                self.system_config = SystemConfigData(debug=sys_config.debug)
            else:
                raise ValueError("Load system config failed, please check database.")

            video_config = db.query(VideoConfig).first()
            if video_config:
                self.video_config = VideoConfigData(
                    rtsp=video_config.rtsp,
                    web_camera=video_config.web_camera,
                    image_height=video_config.image_height,
                    image_width=video_config.image_width,
                    detection_range_start_point=[
                        video_config.detection_range_start_point_x,
                        video_config.detection_range_start_point_y,
                    ],
                    detection_range_end_point=[
                        video_config.detection_range_end_point_x,
                        video_config.detection_range_end_point_y,
                    ],
                )
            else:
                raise ValueError("Load video config failed, please check database.")

            reco_config = db.query(FaceRecognitionConfig).first()
            if reco_config:
                self.reco_config = RecoConfigData(
                    enable=reco_config.enable,
                    set_mode=reco_config.set_mode,  # FIXME: FastAPI 會報錯
                    enable_blink_detection=reco_config.enable_blink_detection,
                    dlib_predictor=reco_config.dlib_predictor_path,
                    dlib_recognition_model=reco_config.dlib_recognition_model_path,
                    face_model=reco_config.face_model,
                    minimum_bounding_box_height=reco_config.minimum_bounding_box_height,
                    minimum_face_detection_score=reco_config.minimum_face_detection_score,
                    eyes_detection_brightness_threshold=reco_config.eyes_detection_brightness_threshold,
                    eyes_detection_brightness_value=[
                        reco_config.eyes_detection_brightness_value_min,
                        reco_config.eyes_detection_brightness_value_max,
                    ],
                    sensitivity=reco_config.sensitivity,
                    consecutive_prediction_intervals=reco_config.consecutive_prediction_intervals_frame,
                )
            else:
                raise ValueError("Load face recognition config failed, please check database.")

        except Exception as e:
            print(f"Error loading configuration from database: {e}")
            raise ValueError("Failed to load configuration from database.")

        finally:
            print("--- Configuration loaded from database ---")
            print(f"System Config: {self.reco_config}")
            db.close()


class FaceAppManager:
    """Manager for FaceApp integration with FastAPI"""

    def __init__(self, connection_manager):
        from .connection_manager import ConnectionManager

        self.connection_manager: ConnectionManager = connection_manager
        self.config_adapter = ConfigAdapter()
        self.running = False
        self.face_app = None
        self.frame_queue = asyncio.Queue(maxsize=10)
        self.log_queue = asyncio.Queue(maxsize=100)
        self.detection_results_queue = Queue()

    def toggle_blink_detection(self):
        """Toggle blink detection in FaceApp"""
        if self.face_app:
            self.face_app.toggle_blink_detection()

    async def run(self):
        """Run face detection in async context"""
        self.running = True

        # Import FaceApp here to avoid circular import
        try:
            from face_detection import FaceApp

        except ImportError as e:
            print(f"Error importing FaceApp: {e}")
            return

        # Create FaceApp instance with our config adapter
        self.face_app = FaceApp(
            mode=RunMode.FASTAPI,
            config_source=self.config_adapter,
            frame_queue=self.frame_queue,
            log_queue=self.log_queue,
            external_detection_queue=self.detection_results_queue,
        )

        # Start face detection in a separate thread
        face_thread = threading.Thread(target=self.face_app.run)
        face_thread.daemon = True  # Allow thread to exit when main program exits
        face_thread.start()

        # Start async tasks for streaming and logging
        await asyncio.gather(self._stream_frames(), self._process_logs(), return_exceptions=True)

    async def stop(self):
        """Stop face detection"""
        self.running = False
        if self.face_app:
            self.face_app.stop()
            self.face_app = None

    async def _stream_frames(self):
        """Stream frames to WebSocket clients"""
        while self.running:
            try:
                # Get frame from queue
                frame_data = await asyncio.wait_for(self.frame_queue.get(), timeout=0.1)

                # Send to all connected clients
                await self.connection_manager.send_frame(frame_data)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error streaming frame: {e}")
                await asyncio.sleep(0.01)

    async def _process_logs(self):
        """Process detection logs"""
        while self.running:
            try:
                # Get log from queue
                log_data = await asyncio.wait_for(self.log_queue.get(), timeout=0.1)

                # Save to database
                await self._save_log_to_db(log_data)

                # Send to WebSocket clients
                await self.connection_manager.send_log(log_data)

                # Post to external server (if configured)
                if not self.config_adapter.system_config.debug:
                    asyncio.create_task(post_log_to_server(log_data))

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error processing log: {e}")
                await asyncio.sleep(0.01)

    async def _save_log_to_db(self, log_data: dict):
        """Save detection log to database"""
        db = SessionLocal()
        try:
            db_log = SystemLogs(
                name=log_data.get("name", ""),
                group=log_data.get("group", ""),
                log_level=log_data.get("log_level", "INFO"),
                message=log_data.get("message", ""),
                timestamp=datetime.utcnow(),
            )
            db.add(db_log)
            db.commit()
        except Exception as e:
            print(f"Error saving log to database: {e}")
            db.rollback()
        finally:
            db.close()
