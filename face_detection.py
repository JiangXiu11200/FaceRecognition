import os
import time
import traceback
from multiprocessing import Process, Queue
from threading import Thread

import cv2
import dlib
import mediapipe as mp
import numpy as np

from package import calculation, config, coordinate_detection, predictor
from package import settings as system_settings
from package import video_capturer


class FaceApp:
    def __init__(self):
        # system configuration
        settings = system_settings.Settings()
        settings.load_setting()
        self.video_config = settings.video_config
        self.sys_config = settings.system_config
        self.reco_config = settings.reco_config
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.coordinate_detection = coordinate_detection.CoordinateDetection(
            self.video_config.detection_range_start_point,
            self.video_config.detection_range_end_point,
            self.reco_config.minimum_face_detection_score,
            self.reco_config.minimum_bounding_box_height,
        )
        self.calculation = calculation.Calculation(self.video_config.image_width, self.video_config.image_height)
        self.predictor = predictor.Predictor(
            self.reco_config.dlib_predictor,
            self.reco_config.dlib_recognition_model,
            self.reco_config.registered_face_descriptor,
            self.reco_config.sensitivity,
        )
        self.fps = 0
        self.fps_count = 0
        # video capture
        self.video_queue = Queue()
        self.signal_queue = Queue()
        video_capture = video_capturer.VideoCapturer(self.video_config.rtsp, self.video_queue, self.signal_queue)
        video_capturer_proc = Process(target=video_capture.get_video)
        video_capturer_proc.start()
        config.logger.info("start system")

    @staticmethod
    def _draw_rectangle(frame: np.ndarray, coordinate: list):
        """Draw bounding box.

        Parameters:
            frame (np.ndarray): The frame.
            coordinate (list): The bounding box coordinates, format is [[x1, y1], [x2, y2]].

        Returns:
            None
        """
        cv2.rectangle(frame, (coordinate[0][0], coordinate[0][1]), (coordinate[1][0], coordinate[1][1]), (0, 255, 0), 2)

    @staticmethod
    def _draw_text(frame: np.ndarray, text: str, coordinate: list, color: tuple):
        """
        Draw text.

        Parameters:
            frame (np.ndarray): The frame.
            text (str): The text.
            coordinate (list): The coordinate.
            color (tuple): The color.

        Returns:
            None
        """
        cv2.putText(frame, text, coordinate, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_dlib_features(face_roi: np.ndarray, feature_coordinates: dlib.rectangle):
        """
        Draw dlib features.

        Parameters:
            face_roi (np.ndarray): The face ROI.
            feature_coordinates (dlib.rectangle): The dlib feature coordinates.

        Returns:
            None
        """
        for i in range(68):
            cv2.circle(face_roi, (feature_coordinates.part(i).x, feature_coordinates.part(i).y), 3, (0, 0, 255), 2)
            cv2.putText(
                face_roi, str(i), (feature_coordinates.part(i).x, feature_coordinates.part(i).y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1
            )
        cv2.imshow("face_roi", face_roi)

    def _draw_result_information(self, frame: np.ndarray, detection_results: bool, blink_state: bool, detection_distance: int):
        face_color = (0, 255, 0) if detection_results else (0, 0, 255)
        eyes_color = (0, 255, 0) if blink_state else (0, 0, 255)
        FaceApp._draw_text(frame, "Eyes detection:", (10, 70), (0, 0, 255))
        FaceApp._draw_text(frame, str(blink_state), (260, 70), eyes_color)
        FaceApp._draw_text(frame, "Face detection:", (10, 110), (0, 0, 255))
        FaceApp._draw_text(frame, str(detection_results), (260, 110), face_color)
        FaceApp._draw_text(frame, "Detection distance:", (10, 150), (0, 0, 255))
        FaceApp._draw_text(frame, str(detection_distance), (320, 150), face_color)

    def _eyes_preprocessing(self, frame: np.ndarray, bounding_eye_left: list, bounding_eye_right: list, threshold_value: int):
        """
        Eyes preprocessing.

        Parameters:
            frame (np.ndarray): The frame.
            bounding_eye_left (list): The left eye bounding box coordinates.
            bounding_eye_right (list): The right eye bounding box coordinates.
            threshold_value (int): The threshold value.

        Returns:
            left_eye_gary (np.ndarray): The left eye grayscale.
            right_eye_gary (np.ndarray): The right eye grayscale
        """
        if self.sys_config.debug:
            FaceApp._draw_rectangle(frame, bounding_eye_left)
            FaceApp._draw_rectangle(frame, bounding_eye_right)
        eye_left_roi = frame[bounding_eye_left[0][1] : bounding_eye_left[1][1], bounding_eye_left[0][0] : bounding_eye_left[1][0]]
        eye_right_roi = frame[bounding_eye_right[0][1] : bounding_eye_right[1][1], bounding_eye_right[0][0] : bounding_eye_right[1][0]]
        # blink detection
        left_eye_gary, right_eye_gary = calculation.Calculation.preprocess_eye_regions(eye_left_roi, eye_right_roi, threshold_value)
        return left_eye_gary, right_eye_gary

    def _fps_counter(self):
        if time.time() - self.start_time >= 1:
            self.fps = self.fps_count
            self.fps_count = 0
            self.start_time = time.time()

    def run(self):
        # fps parameters
        self.start_time = time.time()
        # detection parameters
        face_roi: np.array = None
        eyes_blink: list[list] = [[], []]
        face_in_detection_range: bool = False
        enable_execution_interval: bool = False
        interval_count: int = 0
        detection_results_queue = Queue()
        detection_results: bool = False
        detection_distance: int = 0
        # blink detection parameters
        left_median: int = 1
        right_median: int = 1
        blink_state: bool = False
        blink_count: int = 0
        average_brightness: np.float64 = 0
        threshold_value: int = 0
        with self.mp_face_detection as face_detection:
            while True:
                try:
                    self.fps_count += 1
                    self._fps_counter()
                    if self.video_queue.empty():
                        cv2.waitKey(1)
                    key = cv2.waitKey(1)
                    frame = cv2.resize(
                        self.video_queue.get(), (self.video_config.image_width, self.video_config.image_height), interpolation=cv2.INTER_AREA
                    )
                    FaceApp._draw_rectangle(frame, [self.video_config.detection_range_start_point, self.video_config.detection_range_end_point])
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(frame_bgr)
                    if results.detections:
                        for detection_mp in results.detections:
                            # face bounding box
                            bounding_box_mp = detection_mp.location_data.relative_bounding_box
                            bounding_box_height = round(bounding_box_mp.height, 2)
                            detection_score = round(detection_mp.score[0], 2)
                            face_bounding_box, center = self.calculation.get_face_boundingbox(bounding_box_mp)
                            if not self.coordinate_detection.face_box_in_roi(center, bounding_box_height, detection_score):
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
                                    threshold_value = self.reco_config.eyes_detection_brightness_value[0]
                                else:
                                    threshold_value = self.reco_config.eyes_detection_brightness_value[1]
                                config.logger.debug(f"eyes bounding box average brightness: {average_brightness}")
                            if self.sys_config.debug:
                                FaceApp._draw_rectangle(frame, face_bounding_box)
                        if self.reco_config.enable and average_brightness != 0 and face_in_detection_range:
                            # eyes bounding box
                            bounding_eye_left, bounding_eye_right = self.calculation.get_eyes_boundingbox(detection_mp, bounding_box_mp.height)
                            left_eye_gary, right_eye_gary = self._eyes_preprocessing(frame, bounding_eye_left, bounding_eye_right, threshold_value)
                            if left_eye_gary is not None or right_eye_gary is not None:
                                eyes_blink[0].append((left_eye_gary == 0).sum())
                                eyes_blink[1].append((right_eye_gary == 0).sum())
                            if len(eyes_blink[0]) > 15 and len(eyes_blink[1]) > 15:
                                blink_state, left_median, right_median = calculation.Calculation.blink_detect(
                                    eyes_blink, blink_count, left_median, right_median
                                )
                                if (self.sys_config.debug and key == ord("r") or key == ord("R")) or (
                                    not self.sys_config.debug and blink_state and not enable_execution_interval
                                ):  # If debug mode is enabled, prediction can only be performed by pressing "r"
                                    extraction = Thread(
                                        target=self.predictor.face_prediction,
                                        args=(
                                            face_roi,
                                            detection_results_queue,
                                        ),
                                    )
                                    extraction.start()
                                    enable_execution_interval = True
                                    interval_count = 0
                                if enable_execution_interval:
                                    interval_count += 1
                                    if interval_count >= self.reco_config.consecutive_prediction_intervals:
                                        enable_execution_interval = False
                                        detection_results = False
                                        interval_count = 0
                                if self.sys_config.debug:
                                    cv2.imshow("eyes_left", left_eye_gary)
                                    cv2.imshow("eyes_right", right_eye_gary)
                        if self.reco_config.set_mode and face_roi is not None:
                            if key == ord("s") or key == ord("S"):
                                face_descriptor, feature_coordinates = self.predictor.feature_extraction(face_roi)
                                predictor.Predictor.save_feature(self.reco_config.face_model, face_descriptor)
                                if self.sys_config.debug:
                                    FaceApp._draw_dlib_features(face_roi, feature_coordinates)
                    else:
                        eyes_blink = [[], []]
                        blink_count = 0
                        average_brightness = 0
                        face_roi = None
                    if not detection_results_queue.empty():
                        detection_result = detection_results_queue.get()
                        detection_distance = round(detection_result[1], 2)
                        if detection_result[0]:
                            detection_results = True
                        else:
                            detection_results = False
                    self._draw_result_information(frame, detection_results, blink_state, detection_distance)
                    if self.sys_config.debug:
                        FaceApp._draw_text(frame, "FPS: " + str(self.fps), (10, 30), (0, 0, 255))
                    cv2.imshow("video_out", frame)
                    if key == ord("q") or key == ord("Q"):
                        self.signal_queue.put(1)
                        break
                except Exception as e:
                    traceback.print_exc()
                    self.signal_queue.put(1)
                    time.sleep(1)
                    os.kill(os.getpid(), 9)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = FaceApp()
    app.run()
