import threading
import time
import traceback
from multiprocessing import Queue

import cv2


class VideoCapturer:
    def __init__(self, rtsp: str, video_queue: Queue):
        self.rtsp = rtsp
        self.video_queue = video_queue
        self.stop_event = threading.Event()

    def get_video(self):
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

        except Exception:
            traceback.print_exc()
        finally:
            if self.cap:
                self.cap.release()
            print("VideoCapturer stopped.")

    def stop(self):
        """通知 thread 停止"""
        self.stop_event.set()
