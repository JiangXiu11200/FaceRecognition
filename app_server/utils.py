import base64
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def post_log_to_server(log_data: dict, server_url: str = None):
    """
    Post detection log to external server

    Args:
        log_data: Detection log data
        server_url: External server URL (from environment variable or config)
    """
    if not server_url:
        # Get from environment variable or config
        import os

        server_url = os.getenv("EXTERNAL_LOG_SERVER_URL")

    if not server_url:
        logger.warning("No external log server URL configured")
        return

    try:
        async with aiohttp.ClientSession() as session:
            # FIXME: 需要修正 post 過去的資料格式
            async with session.post(f"{server_url}", json=log_data, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    logger.info("Successfully posted log to external server")
                else:
                    logger.error(f"Failed to post log: {response.status}")
    except Exception as e:
        logger.error(f"Error posting log to external server: {e}")


def save_face_image(face_roi: np.ndarray, success: bool, person_name: Optional[str] = None) -> str:
    """
    Save face image to local directory

    Args:
        face_roi: Face region of interest
        success: Whether detection was successful
        person_name: Name of detected person (if identified)

    Returns:
        Path to saved image
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    status = "success" if success else "failed"

    # Create directory structure
    base_dir = Path("captured_faces")
    date_dir = base_dir / datetime.now().strftime("%Y%m%d")
    date_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    if person_name:
        filename = f"{timestamp}_{status}_{person_name}.jpg"
    else:
        filename = f"{timestamp}_{status}_unknown.jpg"

    filepath = date_dir / filename

    # Save image
    cv2.imwrite(str(filepath), face_roi)
    logger.info(f"Saved face image: {filepath}")

    return str(filepath)


def encode_frame_to_base64(frame: np.ndarray, quality: int = 70) -> str:
    """
    Encode OpenCV frame to base64 string

    Args:
        frame: OpenCV frame
        quality: JPEG quality (0-100)

    Returns:
        Base64 encoded string
    """
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode("utf-8")


def decode_base64_to_frame(base64_string: str) -> np.ndarray:
    """
    Decode base64 string to OpenCV frame

    Args:
        base64_string: Base64 encoded image

    Returns:
        OpenCV frame
    """
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
