import os
import traceback

import cv2


def get_video(video_queue, signal_queue):
    try:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        while ret:
            if not signal_queue.empty():
                break
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            video_queue.put(frame)
            cv2.waitKey(1)
    except Exception as e:
        traceback.print_exc()
    cap.release()
    os._exit(0)