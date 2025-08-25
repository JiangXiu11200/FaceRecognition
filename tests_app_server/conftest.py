from collections.abc import Generator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app_server.db.database import Base, get_db
from app_server.db.models import FaceRecognitionConfig, SystemConfig, VideoConfig
from main import app

SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db_session() -> Generator[Session, None, None]:
    """Create test database session."""
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()

    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """Create test client with overridden database dependency."""

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def sample_config(db_session: Session):
    """Create a sample configuration in the test database."""
    face_config = FaceRecognitionConfig(
        enable=True,
        set_mode=True,
        enable_blink_detection=True,
        dlib_predictor_path="models/dlib/shape_predictor_68_face_landmarks.dat",
        dlib_recognition_model_path="models/dlib/dlib_face_recognition_resnet_model_v1.dat",
        face_model="models/face_recognition/model.csv",
        minimum_bounding_box_height=0.4,
        minimum_face_detection_score=0.6,
        eyes_detection_brightness_threshold=120,
        eyes_detection_brightness_value_min=50,
        eyes_detection_brightness_value_max=20,
        sensitivity=0.4,
        consecutive_prediction_intervals_frame=90,
    )

    system_config = SystemConfig(
        debug=False,
    )

    video_config = VideoConfig(
        rtsp="",
        web_camera=0,
        image_width=1280,
        image_height=720,
        detection_range_start_point_x=420,
        detection_range_start_point_y=160,
        detection_range_end_point_x=820,
        detection_range_end_point_y=560,
    )

    db_session.add(face_config)
    db_session.add(system_config)
    db_session.add(video_config)
    db_session.commit()

    return {"face": face_config, "system": system_config, "video": video_config}


@pytest.fixture
def mock_minio():
    """Mock MinIO 客戶端"""
    with patch("app_server.utils.minio_client.MinioClient") as mock:
        mock.upload_object.return_value = (True, {"message": "success"})
        mock.get_object_url.return_value = (True, {"url": "http://mock-url.com/image.jpg"})
        mock.delete_directory.return_value = (True, {"message": "deleted"})
        yield mock


@pytest.fixture
def sample_base64_image():
    """Sample base64-encoded image string."""
    # 1x1 pixel PNG image
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
