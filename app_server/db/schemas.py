"""
These models are used to validate and serialize data for the FastAPI application.
"""

from pydantic import BaseModel, Field


class SystemConfigBase(BaseModel):
    debug: bool = Field(default=False, description="Enable or disable debug mode")


class VideoConfigBase(BaseModel):
    rtsp: str | None = Field(
        default=None,
        max_length=64,
        description="RTSP stream URL (if not None, use RTSP stream)",
    )
    web_camera: int | None = Field(
        default=None,
        description="Web camera index (if not None, use web camera)",
    )
    image_height: int = Field(default=480, ge=0, le=2160, description="Resize height of the input image")
    image_width: int = Field(default=640, ge=0, le=3840, description="Resize width of the input image")
    detection_range_start_point_x: int = Field(
        default=0, ge=0, le=3840, description="X coordinate of the start point of the detection range"
    )
    detection_range_start_point_y: int = Field(
        default=0, ge=0, le=2160, description="Y coordinate of the start point of the detection range"
    )
    detection_range_end_point_x: int = Field(
        default=0, ge=0, le=3840, description="X coordinate of the end point of the detection range"
    )
    detection_range_end_point_y: int = Field(
        default=0, ge=0, le=2160, description="Y coordinate of the end point of the detection range"
    )


class SystemLogsBase(BaseModel):
    name: str = Field(default="", max_length=64, description="User name")
    group: str = Field(default="", max_length=64, description="User group")
    log_level: str = Field(default="", max_length=16, description="Log level (e.g., INFO, ERROR, DEBUG)")
    message: str = Field(default="", max_length=64, description="Log message")
    timestamp: str = Field(default="", description="Timestamp of the log entry")


class FaceRecognitionConfigBase(BaseModel):
    enable_blink_detection: bool = Field(default=True, description="Enable blink detection")
    dlib_predictor_path: str = Field(default="", max_length=256, description="Path to the Dlib predictor model")
    dlib_recognition_model_path: str = Field(
        default="", max_length=256, description="Path to the Dlib recognition model"
    )
    face_model: str = Field(default="", max_length=256, description="CSV model file for face features")
    minimum_bounding_box_height: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum bounding box height as a ratio (0.0 to 1.0)"
    )
    minimum_face_detection_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum face detection score as a ratio (0.0 to 1.0)"
    )
    eyes_detection_brightness_threshold: int = Field(
        default=0.0, ge=0, le=255, description="Brightness threshold for eyes detection"
    )
    eyes_detection_brightness_value_min: int = Field(
        default=0, ge=0, le=255, description="Minimum brightness value for eyes detection"
    )
    eyes_detection_brightness_value_max: int = Field(
        default=0, ge=0, le=255, description="Maximum brightness value for eyes detection"
    )
    sensitivity: float = Field(default=0.5, ge=0.0, le=1.0, description="Sensitivity for face detection (0.0 to 1.0)")
    consecutive_prediction_intervals_frame: int = Field(
        default=90, ge=10, le=150, description="Number of frames for consecutive predictions (10 to 150)"
    )
