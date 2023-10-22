import cv2
import csv
import dlib
import json
import mediapipe as mp
import numpy as np
import os
import traceback
import time
from multiprocessing import Queue, Process
from video_capturer import video_capturer

class FaceApp():
    def __init__(self):
        with open("../setting.json") as conf:
            conf = json.load(conf)
            self.height = conf["sys_config"]["height"]
            self.width = conf["sys_config"]["width"]
        self.predictor = dlib.shape_predictor("../lib/shape_predictor_68_face_landmarks.dat")
        self.face_reco = dlib.face_recognition_model_v1("../lib/dlib_face_recognition_resnet_model_v1.dat")
        self.video_queue = Queue()
        self.signal_queue = Queue()
        self.mp_face_detection = mp.solutions.face_detection
        video_capturer_proc = Process(target=video_capturer, args=(self.video_queue, self.signal_queue))
        video_capturer_proc.start()

    def run(self):
        fps = 0
        fps_count = 0
        fps_start_time = time.time()
        face_check_count = 0
        with self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            while True:
                try:
                    fps_count += 1
                    if time.time() - fps_start_time >= 1:
                        fps = fps_count
                        fps_count = 0
                        fps_start_time = time.time()
                    if self.video_queue.empty():
                        cv2.waitKey(1)
                    frame = cv2.resize(self.video_queue.get(), (self.width, self.height), interpolation=cv2.INTER_AREA)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(frame_bgr)
                    if results.detections:
                        for detection in results.detections:
                            bounding_box = detection.location_data.relative_bounding_box
                            bounding_x1, bounding_y1 = int(bounding_box.xmin * self.width), int(bounding_box.ymin * self.height)
                            bounding_x2, bounding_y2 = int(bounding_x1 + bounding_box.width * self.width), int(bounding_y1 + bounding_box.height * self.height)
                            cv2.rectangle(frame, (bounding_x1, bounding_y1), (bounding_x2, bounding_y2), (0, 255, 0), 2)
                            face_roi = frame[bounding_y1 : bounding_y2, bounding_x1 : bounding_x2]

                        if face_roi.shape[0] > 150 and face_roi.shape[1] > 150:
                            face_check_count += 1
                            if face_check_count >= 60:
                                self.feature_extraction(face_roi)
                                face_check_count = 0
                            cv2.putText(frame, str(face_check_count), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)                           
                        else:
                            face_check_count = 0

                    cv2.putText(frame, str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow("video_out", frame)
                    key = cv2.waitKey(1)
                    if key == ord("q") or key == ord("Q"):
                        self.signal_queue.put(1)
                        break
                except Exception as e:
                    traceback.print_exc()
                    self.signal_queue.put(1)
                    time.sleep(1)
                    os.kill(os.getpid(), 9)

        cv2.destroyAllWindows()

    def feature_extraction(self, face_roi):
        landmarks_frame = cv2.cvtColor(face_roi, cv2. COLOR_BGR2RGB)
        dlib_coordinate = dlib.rectangle(0, 0, face_roi.shape[0], face_roi.shape[1])
        shape = self.predictor(landmarks_frame, dlib_coordinate)
        face_descriptor = np.array(self.face_reco.compute_face_descriptor(face_roi, shape))
        print("face_descriptor: ", face_descriptor)
        # 繪製dlib特徵點
        # for i in range(68):
        #     cv2.circle(face_roi,(shape.part(i).x, shape.part(i).y), 3, (0, 0, 255), 2)
        #     cv2.putText(face_roi, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
        # cv2.imshow("face_roi", face_roi)

if __name__ == "__main__":
    FaceApp().run()