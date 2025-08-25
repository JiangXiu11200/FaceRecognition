"""
Face Recognition application database models using SQLAlchemy.
"""

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String

from .database import Base


class SystemConfig(Base):
    __tablename__ = "system_config"

    id = Column(Integer, primary_key=True, index=True)
    debug = Column(Boolean, default=False)


class VideoConfig(Base):
    __tablename__ = "video_config"

    id = Column(Integer, primary_key=True, index=True)
    rtsp = Column(String, nullable=True)  # FIXME: If not none, use web camera
    web_camera = Column(Integer, nullable=True)  # FIXME: If not none, use web camera
    image_height = Column(Integer, default=480)
    image_width = Column(Integer, default=640)
    detection_range_start_point_x = Column(Integer, default=0)
    detection_range_start_point_y = Column(Integer, default=0)
    detection_range_end_point_x = Column(Integer, default=0)
    detection_range_end_point_y = Column(Integer, default=0)


class SystemLogs(Base):
    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    group = Column(String, nullable=False)  # The group name obtained from microservice,
    log_level = Column(String, nullable=False)
    message = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class FaceRecognitionConfig(Base):
    __tablename__ = "face_recognition_config"

    id = Column(Integer, primary_key=True, index=True)
    enable = Column(Boolean, default=True)
    set_mode = Column(Boolean, default=False)  # FIXME: FastAPI 會報錯
    enable_blink_detection = Column(Boolean, default=True)
    dlib_predictor_path = Column(String, nullable=True)
    dlib_recognition_model_path = Column(String, nullable=True)
    face_model = Column(
        String, nullable=True
    )  # TODO: Path to the CSV model file for face features, change to key-value db
    minimum_bounding_box_height = Column(Float, default=0.0)
    minimum_face_detection_score = Column(Float, default=0.0)
    eyes_detection_brightness_threshold = Column(Integer, default=0)
    eyes_detection_brightness_value_min = Column(Integer, default=0)
    eyes_detection_brightness_value_max = Column(Integer, default=0)
    sensitivity = Column(Float, default=0.5)
    consecutive_prediction_intervals_frame = Column(Integer, default=90)
