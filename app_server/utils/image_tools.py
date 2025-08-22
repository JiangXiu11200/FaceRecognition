import base64

import cv2
import numpy as np


def base64_to_bgr(base64_str: str) -> tuple[bytes, np.ndarray]:
    if "," in base64_str:  # Remove data URL prefix data:image/jpeg;base64
        base64_str = base64_str.split(",", 1)[1]

    image_bytes = base64.b64decode(base64_str)
    nparr = np.frombuffer(image_bytes, np.uint8)

    img_nparr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_nparr is None or img_nparr.size == 0:
        raise ValueError("Decoded image is empty or invalid")
    return image_bytes, img_nparr
