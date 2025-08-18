import os
import traceback
from multiprocessing import Queue

import cv2


class VideoCapturer:
    def __init__(self, rtsp: str, video_queue: Queue, signal_queue: Queue):
        self.rtsp = rtsp
        self.video_queue = video_queue
        self.signal_queue = signal_queue

    def get_video(self):
        """
        Get video stream. Initialize VideoCapturer and input video queue and signal queue.

        Parameters:
            rtsp (str): The RTSP URL.
            video_queue (Queue): The video queue.
            signal_queue (Queue): Used to interrupt video capture.

        Returns:
            None

        Methods:
            Use video_queue.get() directly to get frames.
        """
        try:
            cap = cv2.VideoCapture(self.rtsp, cv2.CAP_AVFOUNDATION)
            ret, frame = cap.read()
            while ret:
                if not self.signal_queue.empty():
                    break
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                self.video_queue.put(frame)
                cv2.waitKey(1)
        except Exception:
            traceback.print_exc()
        cap.release()
        os._exit(0)
