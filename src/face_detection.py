import os
import time
import traceback
from multiprocessing import Process, Queue
from threading import Thread

import cv2
import mediapipe as mp
import numpy as np
from package.settings import Settings
from package.video_capturer import VideoCapturer
from package import config
import package


class FaceApp(package.calculation):
    def __init__(self):
        self.settings = Settings()
        self.settings.load_setting()
        rtsp = self.settings.video_config.rtsp
        self.video_queue = Queue()
        self.signal_queue = Queue()
        self.mp_face_detection = mp.solutions.face_detection
        video_capture = VideoCapturer(rtsp, self.video_queue, self.signal_queue)
        video_capturer_proc = Process(target=video_capture.get_video)
        video_capturer_proc.start()
        config.logger.info("start system")

    def run(self):
        fps = 0
        fps_count = 0
        fps_start_time = time.time()
        face_roi = None
        eyes_blink = [[], []]
        left_median = 1
        right_median = 1
        blink_state = False
        blink_count = 0
        face_in_roi = False
        average_brightness = 0
        grayscale_value = 0
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
                    frame = cv2.resize(self.video_queue.get(), (self.settings.video_config.image_width, self.settings.video_config.image_height), interpolation=cv2.INTER_AREA)
                    cv2.rectangle(frame, (self.settings.video_config.detection_range_start_point[0], self.settings.video_config.detection_range_start_point[1]), \
                                (self.settings.video_config.detection_range_end_point[0], self.settings.video_config.detection_range_end_point[1]), (0, 255, 0), 2)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(frame_bgr)
                    if results.detections:
                        for detection in results.detections:
                            # face bounding box
                            bounding_box = detection.location_data.relative_bounding_box
                            bounding_box_height = round(bounding_box.height, 2)
                            detection_score = round(detection.score[0], 2)
                            face_box_x1, face_box_y1, face_box_x2, face_box_y2, face_box_center_x, face_box_center_y = self.get_face_boundingbox(bounding_box, self.settings.video_config.image_width, self.settings.video_config.image_height)
                            if self.settings.system_config.debug:
                                cv2.rectangle(frame, (face_box_x1, face_box_y1), (face_box_x2, face_box_y2), (0, 255, 0), 2)
                                cv2.circle(frame, (face_box_center_x, face_box_center_y), 3, (0, 0, 225), -1)
                            if not self.detection_range(self.settings.video_config.detection_range_start_point, self.settings.video_config.detection_range_end_point, face_box_center_x, face_box_center_y , \
                                                        bounding_box_height, detection_score, self.settings.reco_config.minimum_bounding_box_height, self.settings.reco_config.minimum_face_detection_score):
                                average_brightness = 0
                                blink_count = 0
                                eyes_blink = [[], []]
                                face_in_roi = False
                                continue
                            blink_count += 1
                            face_roi = frame[face_box_y1 : face_box_y2, face_box_x1 : face_box_x2]
                            face_in_roi = np.all(np.array(face_roi.shape) != 0)
                            if average_brightness == 0 and face_in_roi and blink_count % 5 == 0:
                                hsv_image = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
                                _, _, value = cv2.split(hsv_image)
                                average_brightness = np.mean(value)
                                if average_brightness > 120:
                                    grayscale_value = 80
                                else:
                                    grayscale_value = 40
                        if self.settings.reco_config.enable and average_brightness != 0 and face_in_roi:
                            # eyes bounding box
                            eyes_left_x1, eyes_left_y1, eyes_left_x2, eyes_left_y2, \
                            eyes_right_x1, eyes_right_y1, eyes_right_x2, eyes_right_y2 = self.get_eyes_boundingbox(detection, bounding_box.height, self.settings.video_config.image_width, self.settings.video_config.image_height)
                            if self.settings.system_config.debug:
                                cv2.rectangle(frame, (eyes_left_x1, eyes_left_y1), (eyes_left_x2, eyes_left_y2), (0, 255, 0), 2)
                                cv2.rectangle(frame, (eyes_right_x1, eyes_right_y1), (eyes_right_x2, eyes_right_y2), (0, 255, 0), 2)
                            eye_left_roi = frame[eyes_left_y1 : eyes_left_y2, eyes_left_x1 : eyes_left_x2]
                            eye_right_roi = frame[eyes_right_y1 : eyes_right_y2, eyes_right_x1 : eyes_right_x2]
                            # blink detection
                            left_eye_gary, right_eye_gary =  self.grayscale_area(eye_left_roi, eye_right_roi, grayscale_value)
                            if self.settings.system_config.debug:
                                cv2.imshow("eyes_left", left_eye_gary)
                                cv2.imshow("eyes_right", right_eye_gary)
                            if left_eye_gary is not None or right_eye_gary is not None:
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
                                left_blink_state = self.easy_blink_detect(left_blink)
                                right_blink_state = self.easy_blink_detect(right_blink)
                                if left_blink_state and right_blink_state:
                                    blink_state = True
                                    cv2.putText(frame, str(blink_state), (eyes_left_x1, eyes_left_x1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                                else:
                                    blink_state = False
                                    cv2.putText(frame, str(blink_state), (eyes_left_x1, eyes_left_x1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                                key = cv2.waitKey(1)
                                if key == ord("r"):
                                    extraction = Thread(target=self.feature_extraction, args=(face_roi, self.settings.reco_config.dlib_predictor, self.settings.reco_config.dlib_recognition_model, self.settings.reco_config.face_features,))
                                    extraction.start()
                        if self.settings.reco_config.set_mode and face_roi is not None:
                            key = cv2.waitKey(1)
                            if key == ord("s"):
                                face_descriptor = self.feature_extraction(face_roi, self.settings.reco_config.dlib_predictor, self.settings.reco_config.dlib_recognition_model, self.settings.reco_config.face_features)
                                self.save_feature(self.settings.reco_config.face_model, face_descriptor)
                    else:
                        blink_count = 0
                        eyes_blink = [[], []]
                        average_brightness = 0
                        face_roi = None
                    cv2.putText(frame, str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow("video_out", frame)
                    key = cv2.waitKey(1)
                    if key == ord("q"):
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

if __name__ == "__main__":
    app = FaceApp()
    app.run()