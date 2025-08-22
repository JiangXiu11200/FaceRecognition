"""
Video streaming for app_server.
"""

import asyncio
import base64
import queue
import threading
import time
from typing import Any, Optional

import cv2
import numpy as np

import package.config as config
import package.video_capturer as video_capturer


class VideoStream:
    def __init__(self, config_source: Optional[Any] = None, frame_queue: Optional[Any] = None):
        self.frame_queue = frame_queue
        self.video_config = config_source.video_config

        self.fps = 0
        self.fps_count = 0

        self.video_queue = queue.Queue()
        video_source = self.video_config.rtsp if self.video_config.rtsp else self.video_config.web_camera
        self.video_capture = video_capturer.VideoCapturer(video_source, self.video_queue)
        self.video_capturer_thread = threading.Thread(target=self.video_capture.get_video)
        self.video_capturer_thread.start()

        self.running = True

        config.logger.debug(f"Started video stream from {video_source}")

    def stop(self):
        self.running = False
        if hasattr(self, "video_capturer_thread") and self.video_capturer_thread.is_alive():
            self.video_capture.stop()
            self.video_capturer_thread.join(timeout=2)

        config.logger.info("Video stream stopped.")

    # TAG: FastAPI mode methods
    async def _put_frame_async(self, frame: np.ndarray):
        """Put frame into queue (FastAPI mode)"""

        try:
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_base64 = base64.b64encode(buffer).decode("utf-8")

            if not self.frame_queue.full():
                await self.frame_queue.put(frame_base64)
        except Exception as e:
            config.logger.error(f"Error putting frame to queue: {e}")

    def _fps_counter(self):
        if time.time() - self.start_time >= 1:
            self.fps = self.fps_count
            self.fps_count = 0
            self.start_time = time.time()

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        self.start_time = time.time()
        while self.running:
            if self.video_capture.stop_event.is_set():
                break

            if not self.video_queue.empty():
                self.fps_count += 1
                self._fps_counter()

                frame = cv2.resize(
                    self.video_queue.get(),
                    (self.video_config.image_width, self.video_config.image_height),
                    interpolation=cv2.INTER_AREA,
                )

                if frame is not None:
                    try:
                        if self.fps_count % 3 == 0:
                            loop.run_until_complete(self._put_frame_async(frame))
                    except Exception as e:
                        config.logger.error(f"Error processing frame: {e}")

        if loop:
            print("Stopping event loop...")
            loop.close()
        self.stop()
