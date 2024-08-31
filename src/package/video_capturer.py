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
        '''
        Get video stream. \n
        Initialize VideoCapturer and input video queue and signal queue.
        
        Methods:
            Use video_queue.get() directly to get frames. \n

        Args:
            self.video_queu (Queue)
            self.signal_queue (Queue)

        Returns:
            None
        '''
        try:
            cap = cv2.VideoCapture(self.rtsp)
            ret, frame = cap.read()
            while ret:
                if not self.signal_queue.empty():
                    break
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                self.video_queue.put(frame)
                cv2.waitKey(1)
        except Exception as e:
            traceback.print_exc()
        cap.release()
        os._exit(0)