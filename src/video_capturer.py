import cv2
import os
import traceback
from multiprocessing import Process, Queue
from threading import Thread

def get_rtsp(video_queue, signal_queue):
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

def video_out(video_queue, signal_queue):
    while True:
        if video_queue.empty():
            cv2.waitKey(1)
        frame = video_queue.get()
        key = cv2.waitKey(1)
        if key == ord("q"):
            signal_queue.put(1)
            break
        cv2.imshow("video_out", frame)

if __name__ == "__main__":
    video_queue = Queue()
    signal_queue = Queue()
    video_capturer_proc = Process(target=video_capturer, args=(video_queue, signal_queue,))
    video_out_thread = Thread(target=video_out, args=(video_queue, signal_queue, ))
    video_capturer_proc.start()
    video_out_thread.start()
