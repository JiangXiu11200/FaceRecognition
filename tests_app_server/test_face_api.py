"""Test face registration and deletion APIs."""

import json
import os
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient


class TestFaceRegistrationAPI:
    """Test Face Registration API."""

    @patch("main.FaceFeatureExtractor")
    def test_register_face_success(
        self, mock_extractor, client: TestClient, sample_config, sample_base64_image, mock_minio
    ):
        # mock the feature extractor methods
        mock_instance = Mock()
        mock_instance.get_face_roi.return_value = (True, {"face_roi": "mock_roi"})
        mock_instance.feature_extraction.return_value = (True, {"face_descriptor": "mock_descriptor"})
        mock_instance.save_feature.return_value = (True, {"message": "saved"})
        mock_extractor.return_value = mock_instance

        # Do not upload to S3
        with patch.dict(os.environ, {"UPLOAD_TO_S3": "false"}):
            response = client.post(
                "/api/register-face", data={"base64_face_image": sample_base64_image, "name": "test_user"}
            )

        assert response.status_code == 201
        data = json.loads(response.content)
        assert data["status"] == "success"
        assert data["registr_name"] == "test_user"

    def test_register_face_no_config(self, client: TestClient, sample_base64_image):
        response = client.post(
            "/api/register-face", data={"base64_face_image": sample_base64_image, "name": "test_user"}
        )

        assert response.status_code == 404
        assert "configuration not found" in response.json()["detail"].lower()

    def test_register_face_missing_data(self, client: TestClient, sample_config):
        # missing name
        response = client.post("/api/register-face", data={"base64_face_image": "some_image"})
        assert response.status_code == 422

        # missing image
        response = client.post("/api/register-face", data={"name": "test_user"})
        assert response.status_code == 422

    @patch("main.FaceFeatureExtractor")
    def test_register_face_extraction_failure(
        self, mock_extractor, client: TestClient, sample_config, sample_base64_image
    ):
        # test face extraction failure
        mock_instance = Mock()
        mock_instance.get_face_roi.return_value = (False, {"error": "No face detected"})
        mock_extractor.return_value = mock_instance

        response = client.post(
            "/api/register-face", data={"base64_face_image": sample_base64_image, "name": "test_user"}
        )

        assert response.status_code in [500, 503]

    def test_delete_face_success(self, client: TestClient, sample_config):
        with patch("main.FaceFeatureExtractor.delete_feature") as mock_delete:
            mock_delete.return_value = (True, {"message": "deleted"})

            response = client.post("/api/delete-registered-face/test_user")

            assert response.status_code == 200
            data = json.loads(response.content)
            assert data["message"] == "deleted"

    def test_delete_face_not_found(self, client: TestClient, sample_config):
        with patch("main.FaceFeatureExtractor.delete_feature") as mock_delete:
            mock_delete.return_value = (False, {"error": "User not found"})

            response = client.post("/api/delete-registered-face/nonexistent_user")

            assert response.status_code == 404
