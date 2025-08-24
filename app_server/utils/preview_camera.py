import cv2


async def capture_image_from_camera(camera_source: str | int, resize: tuple[int, int]) -> bytes:
    try:
        cap = cv2.VideoCapture(camera_source)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera with index {camera_source}")

        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("Failed to capture image from camera")

        frame = cv2.resize(
            frame,
            resize,
            interpolation=cv2.INTER_AREA,
        )
        frame_bytes = cv2.imencode(".jpg", frame)[1].tobytes()

        cap.release()

        return frame_bytes
    except Exception as e:
        raise RuntimeError(f"Error capturing image from camera: {e}")
