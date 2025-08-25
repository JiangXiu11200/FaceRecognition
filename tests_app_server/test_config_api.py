"""Config API tests."""

from fastapi.testclient import TestClient


class TestFaceRecognitionConfigAPI:
    """Test Face Recognition Configuration API."""

    def test_get_config_success(self, client: TestClient, sample_config):
        """Test retrieving configuration."""
        response = client.get("/api/face-reco-config")

        assert response.status_code == 200
        data = response.json()
        assert data["minimum_bounding_box_height"] == 0.4
        assert data["minimum_face_detection_score"] == 0.6

    def test_get_config_not_found(self, client: TestClient):
        """Test confuguration not found."""
        response = client.get("/api/face-reco-config")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_update_config_success(self, client: TestClient, sample_config):
        """Test updating configuration."""
        update_data = {
            "enable_blink_detection": False,
            "eyes_detection_brightness_threshold": 120,
            "eyes_detection_brightness_value_min": 50,
            "eyes_detection_brightness_value_max": 20,
        }

        response = client.post("/api/face-reco-config", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert not data["enable_blink_detection"]
        assert data["eyes_detection_brightness_threshold"] == 120
        assert data["eyes_detection_brightness_value_min"] == 50
        assert data["eyes_detection_brightness_value_max"] == 20

    def test_create_config_success(self, client: TestClient):
        """Test creating configuration."""
        config_data = {
            "enable_blink_detection": True,
            "dlib_predictor_path": "models/dlib/shape_predictor_68_face_landmarks.dat",
            "dlib_recognition_model_path": "models/dlib/dlib_face_recognition_resnet_model_v1.dat",
            "face_model": "models/face_recognition/model.csv",
            "minimum_bounding_box_height": 0.3,
            "minimum_face_detection_score": 0.5,
            "eyes_detection_brightness_threshold": 100,
            "eyes_detection_brightness_value_min": 40,
            "eyes_detection_brightness_value_max": 30,
            "sensitivity": 0.3,
            "consecutive_prediction_intervals_frame": 80,
        }

        response = client.post("/api/face-reco-config", json=config_data)

        assert response.status_code == 200
        data = response.json()
        assert data["eyes_detection_brightness_threshold"] == 100

    def test_update_config_invalid_data(self, client: TestClient):
        """Test updating configuration with invalid data."""
        invalid_data = {
            "eyes_detection_brightness_threshold": "",  # 應該是浮點數
        }

        response = client.post("/api/face-reco-config", json=invalid_data)
        assert response.status_code == 422


class TestSystemConfigAPI:
    """Test System Configuration API."""

    def test_debug_info_success(self, client: TestClient, sample_config):
        response = client.get("/api/debug")

        assert response.status_code == 200
        data = response.json()
        assert data["debug"] is False

    def test_update_debug_info(self, client: TestClient, sample_config):
        update_data = {"debug": True}

        response = client.post("/api/debug", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["debug"] is True


class TestVideoConfigAPI:
    """Test Video Configuration API."""

    def test_get_video_config_success(self, client: TestClient, sample_config):
        response = client.get("/api/video-config")

        assert response.status_code == 200
        data = response.json()
        assert data["rtsp"] == ""
        assert data["web_camera"] == 0
        assert data["detection_range_end_point_y"] == 560

    def test_get_video_config_not_found(self, client: TestClient):
        response = client.get("/api/video-config")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_update_video_config_success(self, client: TestClient, sample_config):
        update_data = {
            "rtsp": "rtsp://test_stream:554/stream",
            "image_width": 800,
            "image_height": 600,
        }

        response = client.post("/api/video-config", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["rtsp"] == "rtsp://test_stream:554/stream"
        assert data["image_width"] == 800
        assert data["image_height"] == 600

    def test_create_video_config_success(self, client: TestClient):
        config_data = {
            "rtsp:": "",
            "web_camera": 1,
            "image_width": 800,
            "image_height": 600,
            "detection_range_start_point_x": 100,
            "detection_range_start_point_y": 100,
            "detection_range_end_point_x": 700,
            "detection_range_end_point_y": 500,
        }

        response = client.post("/api/video-config", json=config_data)

        assert response.status_code == 200
        data = response.json()
        assert data["web_camera"] == 1
        assert data["image_height"] == 600
        assert data["detection_range_end_point_x"] == 700

    def test_update_video_config_invalid_data(self, client: TestClient):
        invalid_data = {
            "image_width": -100,
        }

        response = client.post("/api/video-config", json=invalid_data)
        assert response.status_code == 422
