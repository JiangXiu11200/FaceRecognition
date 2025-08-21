import threading
import time
from multiprocessing import Queue

import cv2

import package.config as config


class VideoCapturer:
    """
    Video Capturer class to capture video frames from RTSP or webcam. \n
    Though `vide_queue` to store frames and `stop_event` to signal when to stop capturing.
    """

    def __init__(self, rtsp: str, video_queue: Queue):
        self.rtsp = rtsp
        self.video_queue = video_queue
        self.stop_event = threading.Event()

    def get_video(self) -> None:
        """
        Get video stream. Initialize VideoCapturer and input video queue and signal queue.

        Parameters:
            rtsp (str): The RTSP URL.
            video_queue (Queue): The video queue.

        Returns:
            None

        Methods:
            Use video_queue.get() directly to get frames.
        """
        try:
            self.cap = cv2.VideoCapture(self.rtsp, cv2.CAP_AVFOUNDATION)
            if not self.cap.isOpened():
                print(f"Cannot open camera: {self.rtsp}")
                return

            while not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    print("No frame received, breaking...")
                    break
                frame = cv2.flip(frame, 1)
                self.video_queue.put(frame)
                time.sleep(0.001)
        except BrokenPipeError as e:
            config.logger.debug(f"WebSocket connection broken in VideoCapturer: {e}")
            config.logger.info("Video capturer closed unexpectedly.")
        except Exception as e:
            config.logger.debug(f"Error in VideoCapturer: {e}")
            config.logger.info("Video capturer closed unexpectedly.")
        finally:
            if self.cap:
                self.cap.release()
            config.logger.debug("Video capturer thread stopped.")

    def stop(self):
        self.stop_event.set()
