import os
import time
import traceback
from multiprocessing import Process, Queue
from threading import Thread

import cv2
import mediapipe as mp
import numpy as np

import package
from package import config
from package.settings import Settings
from package.video_capturer import VideoCapturer


class FaceApp(package.calculation):
    def __init__(self):
        # system configuration
        settings = Settings()
        settings.load_setting()
        self.video_config = settings.video_config
        self.sys_config = settings.system_config
        self.reco_config = settings.reco_config
        self.mp_face_detection = mp.solutions.face_detection
        # video capture
        self.video_queue = Queue()
        self.signal_queue = Queue()
        video_capture = VideoCapturer(self.video_config.rtsp, self.video_queue, self.signal_queue)
        video_capturer_proc = Process(target=video_capture.get_video)
        video_capturer_proc.start()
        config.logger.info("start system")

    def draw_rectangle(self, frame: np.ndarray, coordinate: list):
        cv2.rectangle(frame, (coordinate[0][0], coordinate[0][1]), (coordinate[1][0], coordinate[1][1]), (0, 255, 0), 2)

    def blink_detect(self, eyes_blink: list, blink_count: int, left_median: int, right_median: int):
        '''
        Detect blinking of both eyes.

        Args:
            eyes_blink (list): The eyes blink list.
            blink_count (int): The data length of the eyes_blink.
            left_median (int): The left eye median.
            right_median (int): The right eye median.

        Returns:
            blink_state (bool): The blink state.
            left_median (int): The left eye median.
            right_median (int): The right eye median.
        '''
        try:
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
            if left_blink_state and right_blink_state:  # both eyes blink
                blink_state = True
            else:
                blink_state = False
            return blink_state, left_median, right_median
        except Exception as err:
            print(f"blink_detect error: {err}")
            return None

    def run(self):
        # fps parameters
        fps = 0
        fps_count = 0
        start_time = time.time()
        # detection parameters
        face_roi = None
        eyes_blink = [[], []]
        face_in_detection_range = False
        # blink detection parameters
        left_median = 1
        right_median = 1
        blink_state = False
        blink_count = 0
        average_brightness = 0
        grayscale_value = 0
        with self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            while True:
                try:
                    fps_count += 1
                    if time.time() - start_time >= 1:
                        fps = fps_count
                        fps_count = 0
                        start_time = time.time()
                    if self.video_queue.empty():
                        cv2.waitKey(1)
                    frame = cv2.resize(self.video_queue.get(), (self.video_config.image_width, self.video_config.image_height), interpolation=cv2.INTER_AREA)
                    self.draw_rectangle(frame, [[self.video_config.detection_range_start_point[0], self.video_config.detection_range_start_point[1]], \
                                        [self.video_config.detection_range_end_point[0], self.video_config.detection_range_end_point[1]]])
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(frame_bgr)
                    if results.detections:
                        for detection_mp in results.detections:
                            # face bounding box
                            bounding_box_mp = detection_mp.location_data.relative_bounding_box
                            bounding_box_height = round(bounding_box_mp.height, 2)
                            detection_score = round(detection_mp.score[0], 2)
                            face_bounding_box, center = self.get_face_boundingbox(bounding_box_mp, self.video_config.image_width, self.video_config.image_height)
                            if not self.detection_range(self.video_config.detection_range_start_point, self.video_config.detection_range_end_point, center[0], center[1], \
                                                        bounding_box_height, detection_score, self.reco_config.minimum_bounding_box_height, self.reco_config.minimum_face_detection_score):
                                eyes_blink = [[], []]
                                face_in_detection_range = False
                                blink_count = 0
                                average_brightness = 0
                                continue
                            blink_count += 1
                            face_roi = frame[face_bounding_box[0][1] : face_bounding_box[1][1], face_bounding_box[0][0] : face_bounding_box[1][0]]
                            face_in_detection_range = np.all(np.array(face_roi.shape) != 0)
                            if average_brightness == 0 and face_in_detection_range and blink_count % 5 == 0:
                                hsv_image = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
                                _, _, value = cv2.split(hsv_image)
                                average_brightness = np.mean(value)
                                if average_brightness > self.reco_config.eyes_detection_brightness_threshold:
                                    grayscale_value = self.reco_config.eyes_detection_brightness_value[0]
                                else:
                                    grayscale_value = self.reco_config.eyes_detection_brightness_value[1]
                            if self.sys_config.debug:
                                cv2.rectangle(frame, (face_bounding_box[0][0], face_bounding_box[0][1]), (face_bounding_box[1][0], face_bounding_box[1][1]), (0, 255, 0), 2)
                                cv2.circle(frame, (center[0], center[1]), 3, (0, 0, 225), -1)
                        if self.reco_config.enable and average_brightness != 0 and face_in_detection_range:
                            # eyes bounding box
                            bounding_eye_left, bounding_eye_right = self.get_eyes_boundingbox(detection_mp, bounding_box_mp.height, self.video_config.image_width, self.video_config.image_height)
                            if self.sys_config.debug:
                                self.draw_rectangle(frame, bounding_eye_left)
                                self.draw_rectangle(frame, bounding_eye_right)
                            eye_left_roi = frame[bounding_eye_left[0][1] : bounding_eye_left[1][1], bounding_eye_left[0][0] : bounding_eye_left[1][0]]
                            eye_right_roi = frame[bounding_eye_right[0][1] : bounding_eye_right[1][1], bounding_eye_right[0][0] : bounding_eye_right[1][0]]
                            # blink detection
                            left_eye_gary, right_eye_gary =  self.grayscale_area(eye_left_roi, eye_right_roi, grayscale_value)
                            if left_eye_gary is not None or right_eye_gary is not None:
                                eyes_blink[0].append((left_eye_gary == 0).sum())
                                eyes_blink[1].append((right_eye_gary == 0).sum())
                            if len(eyes_blink[0]) > 15 and len(eyes_blink[1]) > 15:
                                blink_state, left_median, right_median = self.blink_detect(eyes_blink, blink_count, left_median, right_median)
                                if self.sys_config.debug:
                                    color = (0, 255, 0) if blink_state else (0, 0, 255)
                                    cv2.putText(frame, str(blink_state), (bounding_eye_left[0][0], bounding_eye_left[0][0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                                    cv2.imshow("eyes_left", left_eye_gary)
                                    cv2.imshow("eyes_right", right_eye_gary)
                                key = cv2.waitKey(1)
                                if key == ord("r"):
                                    extraction = Thread(target=self.feature_extraction, args=(face_roi, self.reco_config.dlib_predictor, self.reco_config.dlib_recognition_model, \
                                                        self.reco_config.face_features,))
                                    extraction.start()
                        if self.reco_config.set_mode and face_roi is not None:
                            key = cv2.waitKey(1)
                            if key == ord("s"):
                                face_descriptor = self.feature_extraction(face_roi, self.reco_config.dlib_predictor, self.reco_config.dlib_recognition_model, self.reco_config.face_features)
                                self.save_feature(self.reco_config.face_model, face_descriptor)
                    else:
                        eyes_blink = [[], []]
                        blink_count = 0
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