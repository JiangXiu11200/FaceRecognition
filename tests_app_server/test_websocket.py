"""WebSocket connection tests."""

import json
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


class TestWebSocket:
    """WebSocket connection tests."""

    def test_websocket_connect(self, client: TestClient):
        with client.websocket_connect("/ws") as websocket:
            websocket.send_text(json.dumps({"type": "ping"}))

            data = websocket.receive_text()
            response = json.loads(data)
            assert response["type"] == "pong"

    @patch("app_server.connection_manager.ConnectionManager.start_face_detection")
    def test_start_detection(self, mock_start, client: TestClient):
        mock_start.return_value = AsyncMock()

        with client.websocket_connect("/ws") as websocket:
            websocket.send_text(json.dumps({"type": "start_detection"}))
            import time

            time.sleep(0.1)

            mock_start.assert_called()

    def test_websocket_invalid_message(self, client: TestClient):
        with client.websocket_connect("/ws") as websocket:
            websocket.send_text("invalid json")

            data = websocket.receive_text()
            message = json.loads(data)
            assert message["type"] == "error"
            assert "Invalid JSON" in message["message"]

    def test_websocket_unknown_type(self, client: TestClient):
        with client.websocket_connect("/ws") as websocket:
            websocket.send_text(json.dumps({"type": "unknown_type"}))

            data = websocket.receive_text()
            message = json.loads(data)
            assert message["type"] == "error"
            assert "Unknown message type" in message["message"]

            websocket.send_text(json.dumps({"type": "ping"}))
            data = websocket.receive_text()
            response = json.loads(data)
            assert response["type"] == "pong"
