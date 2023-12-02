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
from func import calculation, reco
import config

class FaceApp():
    def __init__(self):
        self.height = config.height
        self.width = config.width
        # self.predictor = dlib.shape_predictor("../lib/shape_predictor_68_face_landmarks.dat")
        # self.face_reco = dlib.face_recognition_model_v1("../lib/dlib_face_recognition_resnet_model_v1.dat")
        self.video_queue = Queue()
        self.signal_queue = Queue()
        self.mp_face_detection = mp.solutions.face_detection
        video_capturer_proc = Process(target=video_capturer, args=(self.video_queue, self.signal_queue))
        video_capturer_proc.start()

    def run(self):
        fps = 0
        fps_count = 0
        fps_start_time = time.time()
        # face_check_count = 0
        eyes_blink = [[], []]
        left_median = 1
        right_median = 1
        blink_state = False
        blink_count = 0
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
                            if bounding_box.height < 0.45:
                                blink_count = 0
                                eyes_blink = [[], []]
                                continue
                            blink_count += 1
                            # face bounding box
                            face_box_x1, face_box_y1, face_box_x2, face_box_y2 = calculation.get_face_boundingbox(frame, bounding_box, self.width, self.height)
                            face_roi = frame[face_box_y1 : face_box_y2, face_box_x1 : face_box_x2]
                            # eyes bounding box
                            eyes_left_x1, eyes_left_y1, eyes_left_x2, eyes_left_y2, \
                            eyes_right_x1, eyes_right_y1, eyes_right_x2, eyes_right_y2 = calculation.get_eyes_boundingbox(frame, detection, bounding_box.height, self.width, self.height)
                            eye_left_roi = frame[eyes_left_y1 : eyes_left_y2, eyes_left_x1 : eyes_left_x2]
                            eye_right_roi = frame[eyes_right_y1 : eyes_right_y2, eyes_right_x1 : eyes_right_x2]
                            # blink detection
                            left_eye_gary, right_eye_gary =  calculation.grayscale_area(eye_left_roi, eye_right_roi)
                            eyes_blink[0].append((left_eye_gary == 0).sum())
                            eyes_blink[1].append((right_eye_gary == 0).sum())
                        if len(eyes_blink[0]) > 15 and len(eyes_blink[1]) > 15:
                            eyes_blink[0].pop(0)
                            eyes_blink[1].pop(0)
                            left_blink = np.array(eyes_blink[0])
                            right_blink = np.array(eyes_blink[1])
                            if blink_count % 15 == 0:
                                left_median = int(np.median(eyes_blink[0]) * 0.8)
                                right_median = int(np.median(eyes_blink[1]) * 0.8)
                                blink_state = False
                            left_blink = (left_blink > left_median).astype(int)
                            right_blink = (right_blink > right_median).astype(int)
                            left_state = reco.easy_blink_detect(left_blink)
                            right_state = reco.easy_blink_detect(right_blink)
                            if left_state == True or right_state == True:
                                blink_state = True
                                cv2.putText(frame, str(blink_state), (eyes_left_x1, eyes_left_x1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                            else:
                                blink_state = False
                                cv2.putText(frame, str(blink_state), (eyes_left_x1, eyes_left_x1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                            
                            print(left_blink, " ", right_blink, " left: ", left_state, "  right: ",  right_state, " result: ", blink_state)

                        # if face_roi.shape[0] > 150 and face_roi.shape[1] > 150:
                        #     face_check_count += 1
                        #     if face_check_count >= 60:
                        #         self.feature_extraction(face_roi)
                        #         face_check_count = 0
                        #     cv2.putText(frame, str(face_check_count), (10, 60), cv2.F ONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)                           
                        # else:
                        #     face_check_count = 0
                    else:
                        blink_count = 0
                        eyes_blink = [[], []]
                    cv2.putText(frame, str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow("video_out", frame)
                    key = cv2.waitKey(1)
                    if key == ord("q") or key == ord("Q"):
                        self.signal_queue.put(1)
                        break
                    if key == ord(" "):
                        cv2.waitKey(0)
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
        self.save_csv(face_descriptor)
        # 繪製dlib特徵點
        # for i in range(68):
        #     cv2.circle(face_roi,(shape.part(i).x, shape.part(i).y), 3, (0, 0, 255), 2)
        #     cv2.putText(face_roi, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
        # cfv2.imshow("face_roi", face_roi)

if __name__ == "__main__":
    FaceApp().run()
