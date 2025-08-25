"""Performance tests."""

import statistics
import time

import pytest
from fastapi.testclient import TestClient


class TestPerformance:
    """Test performance of key API endpoints."""

    def test_health_check_response_time(self, client: TestClient):
        """Test health check response time."""
        response_times = []

        for _ in range(100):
            start = time.time()
            response = client.get("/api/health")
            end = time.time()

            assert response.status_code == 200
            response_times.append(end - start)

        avg_time = statistics.mean(response_times)
        p95_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile

        # avg time should be less than 100ms
        assert avg_time < 0.1
        # 95% time should be less than 200ms
        assert p95_time < 0.2

    def test_config_update_performance(self, client: TestClient, sample_config):
        """Test configuration update performance."""
        update_times = []

        for i in range(100):
            config_data = {
                "face_model": f"model_{i}.csv",
                "tolerance": 0.3 + (i * 0.01),
                "frame_skip": i % 10 + 1,
                "dlib_predictor_path": "predictor.dat",
                "dlib_recognition_model_path": "recognition.dat",
            }

            start = time.time()
            response = client.post("/api/face-reco-config", json=config_data)
            end = time.time()

            assert response.status_code == 200
            update_times.append(end - start)

        avg_time = statistics.mean(update_times)
        # avg time should be less than 200ms
        assert avg_time < 0.2

    @pytest.mark.slow
    def test_memory_leak(self, client: TestClient):
        """Test for memory leaks over multiple requests."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Make 1000 requests to health endpoint
        for _ in range(1000):
            client.get("/api/health")

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be less than 50MB
        assert memory_increase < 50
