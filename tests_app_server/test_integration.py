"""
Integration tests.
Test registering a face, deleting a face, and checking the health endpoint.
"""

from unittest.mock import Mock, patch

from fastapi.testclient import TestClient


class TestIntegration:
    def test_complete_face_registration_flow(self, client: TestClient, sample_config, sample_base64_image):
        # 1. Health check
        response = client.get("/api/health")
        assert response.status_code == 200

        # 2. Get face recognition config
        response = client.get("/api/face-reco-config")
        assert response.status_code == 200

        # 3. Register a face
        with patch("main.FaceFeatureExtractor") as mock_extractor:
            mock_instance = Mock()
            mock_instance.get_face_roi.return_value = (True, {"face_roi": "mock"})
            mock_instance.feature_extraction.return_value = (True, {"face_descriptor": "mock"})
            mock_instance.save_feature.return_value = (True, {"message": "saved"})
            mock_extractor.return_value = mock_instance

            response = client.post(
                "/api/register-face", data={"base64_face_image": sample_base64_image, "name": "integration_test_user"}
            )
            assert response.status_code == 201

        # 4. Delete the registered face
        with patch("main.FaceFeatureExtractor.delete_feature") as mock_delete:
            mock_delete.return_value = (True, {"message": "deleted"})

            response = client.post("/api/delete-registered-face/integration_test_user")
            assert response.status_code == 200
