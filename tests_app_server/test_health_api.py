"""Health API tests."""

import pytest
from fastapi.testclient import TestClient


class TestHealthAPI:
    def test_health_check(self, client: TestClient):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "face_detection_running": 0,
            "active_connections": 0,
        }

    def test_health_check_response_format(self, client: TestClient):
        response = client.get("/api/health")
        data = response.json()

        assert isinstance(data["face_detection_running"], bool)
        assert isinstance(data["active_connections"], int)
        assert data["active_connections"] >= 0

    def test_health_check_method_not_allowed(self, client: TestClient):
        response = client.post("/api/health")
        assert response.status_code == 405

        response = client.put("/api/health")
        assert response.status_code == 405

        response = client.delete("/api/health")
        assert response.status_code == 405

    @pytest.mark.parametrize(
        "invalid_path",
        [
            "/api/healthy",
            "/api/healthcheck",
            "/health",
        ],
    )
    def test_health_check_invalid_paths(self, client: TestClient, invalid_path: str):
        response = client.get(invalid_path)
        assert response.status_code == 404
