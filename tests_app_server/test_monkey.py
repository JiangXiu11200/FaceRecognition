"""Monkey tests."""

import random
import string

import pytest
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st


class TestMonkey:
    """Test Monkey"""

    @pytest.mark.parametrize("endpoint", ["/api/face-reco-config", "/api/debug", "/api/video-config"])
    def test_random_json_payload(self, client: TestClient, endpoint: str):
        """Test random JSON payloads to various endpoints."""
        for _ in range(10):
            random_data = {
                "".join(random.choices(string.ascii_letters, k=10)): random.choice(
                    [
                        random.randint(-1000, 1000),
                        random.random(),
                        "".join(random.choices(string.printable, k=50)),
                        None,
                        True,
                        False,
                        [],
                        {},
                    ]
                )
                for _ in range(random.randint(1, 10))
            }

            response = client.post(endpoint, json=random_data)

            assert response.status_code in [200, 400, 422, 404, 500]

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(name=st.text(min_size=0, max_size=1000), image=st.text(min_size=0, max_size=10000))
    def test_register_face_fuzz(self, client: TestClient, name: str, image: str):
        response = client.post("/api/register-face", data={"name": name, "base64_face_image": image})

        assert response.status_code in [200, 201, 400, 404, 422, 500, 503]

    def test_massive_concurrent_requests(self, client: TestClient):
        import concurrent.futures

        def make_request():
            return client.get("/api/health")

        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(make_request) for _ in range(100)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert all(r.status_code == 200 for r in results)

    @pytest.mark.parametrize("method", ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
    def test_random_http_methods(self, client: TestClient, method: str):
        response = client.request(method, "/api/health")

        assert response.status_code in [200, 405, 501]

    def test_malformed_urls(self, client: TestClient):
        malformed_urls = [
            "/api/../../etc/passwd",
            "/api/health%00.json",
            "/api/health/../../../",
            "/api/" + "x" * 10000,
        ]

        for url in malformed_urls:
            response = client.get(url)

            assert response.status_code in [400, 404, 414, 422]
