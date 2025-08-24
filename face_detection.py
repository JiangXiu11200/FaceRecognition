import asyncio
import base64
import hashlib
import threading
import time
from datetime import datetime
from enum import Enum
from multiprocessing import Queue
from pathlib import Path
from typing import Any, Optional

import cv2
import dlib
import mediapipe as mp
import numpy as np

from package import calculation, config, coordinate_detection, predictor, video_capturer
from package import settings as system_settings
from package.blink_detector import BlinkDetector


class RunMode(Enum):
    STANDALONE = "standalone"
    FASTAPI = "fastapi"


class FaceApp:
    def __init__(
        self,
        mode: RunMode = RunMode.STANDALONE,
        config_source: Optional[Any] = None,
        frame_queue: Optional[Any] = None,
        log_queue: Optional[Any] = None,
        external_detection_queue: Optional[Queue] = None,
    ):
        """
        Initialize FaceApp.

        Args:
            mode: startup mode (RunMode.STANDALONE or RunMode.FASTAPI)
            config_source: Optional source for configuration data (if None, uses system_settings)
            frame_queue: FastAPI mode streaming frame queue
            log_queue: FastAPI mode logging queue
            external_detection_queue: Optional external queue for detection results (if None, creates a new Queue)
        """
        self.mode = mode
        self.frame_queue = frame_queue
        self.log_queue = log_queue
        self.running = True
        self._minio_client = None

        # Initialize config adapter
        if mode == RunMode.STANDALONE or config_source is None:
            settings = system_settings.Settings()
            settings.load_setting()
            self.video_config = settings.video_config
            self.sys_config = settings.system_config
            self.reco_config = settings.reco_config
        else:
            self.video_config = config_source.video_config
            self.sys_config = config_source.system_config
            self.reco_config = config_source.reco_config

        # MediaPipe face detection
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )

        # Coordinate detection
        self.coordinate_detection = coordinate_detection.CoordinateDetection(
            self.video_config.detection_range_start_point,
            self.video_config.detection_range_end_point,
            self.reco_config.minimum_face_detection_score,
            self.reco_config.minimum_bounding_box_height,
        )

        # Calculation module
        self.calculation = calculation.Calculation(self.video_config.image_width, self.video_config.image_height)

        # Face predictor
        self.predictor = predictor.Predictor(
            self.reco_config.dlib_predictor,
            self.reco_config.dlib_recognition_model,
            self.reco_config.registered_face_descriptor,
            self.reco_config.sensitivity,
        )

        # Initialize BlinkDetector (Can be controlled via configuration or parameters)
        enable_blink = getattr(self.reco_config, "enable_blink_detection", True)
        self.blink_detector = BlinkDetector(enabled=enable_blink)

        # FPS counter
        self.fps = 0
        self.fps_count = 0

        # Video capture
        self.video_queue = Queue()
        self.video_capture_status_alive = True

        # External detection queue
        self.detection_results_queue = external_detection_queue or Queue()

        # FIXME: rtsp 或 web_camera 資料型態不一致，需統一資料型態處理方法
        video_source = self.video_config.rtsp if self.video_config.rtsp else self.video_config.web_camera
        self.video_capture = video_capturer.VideoCapturer(
            video_source, self.video_queue, self.video_capture_status_alive
        )
        self.video_capturer_thread = threading.Thread(target=self.video_capture.get_video)
        self.video_capturer_thread.start()

        config.logger.info(f"Started FaceApp in {mode.value} mode")
        config.logger.info(f"Blink detection enabled: {self.blink_detector.enabled}")

    @property
    def minio_client(self):
        if self._minio_client is None:
            from app_server.utils.minio_client import MinioClient

            self._minio_client = MinioClient
        return self._minio_client

    def stop(self):
        self.running = False
        self.video_queue.close()
        self.video_queue.join_thread()
        if hasattr(self, "video_capturer_thread") and self.video_capturer_thread.is_alive():
            self.video_capture.stop()
            self.video_capturer_thread.join(timeout=2)

        config.logger.info("FaceApp stopped")

    @staticmethod
    def _draw_rectangle(frame: np.ndarray, coordinate: list) -> None:
        """Draw bounding box."""
        cv2.rectangle(frame, (coordinate[0][0], coordinate[0][1]), (coordinate[1][0], coordinate[1][1]), (0, 255, 0), 2)

    @staticmethod
    def _draw_text(frame: np.ndarray, text: str, coordinate: list, color: tuple):
        """Draw text."""
        cv2.putText(frame, text, coordinate, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_dlib_features(face_roi: np.ndarray, feature_coordinates: dlib.rectangle) -> None:
        """Draw dlib features."""
        for i in range(68):
            cv2.circle(face_roi, (feature_coordinates.part(i).x, feature_coordinates.part(i).y), 3, (0, 0, 255), 2)
            cv2.putText(
                face_roi,
                str(i),
                (feature_coordinates.part(i).x, feature_coordinates.part(i).y),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

    def _draw_result_information(
        self, frame: np.ndarray, detection_results: bool, blink_state: bool, detection_distance: int
    ) -> None:
        """Draw detection results and blink state on the frame."""
        face_color = (0, 255, 0) if detection_results else (0, 0, 255)

        # Draw blink detection information
        if self.blink_detector.enabled:
            eyes_color = (0, 255, 0) if blink_state else (0, 0, 255)
            FaceApp._draw_text(frame, "Eyes detection:", (10, 70), (0, 0, 255))
            FaceApp._draw_text(frame, str(blink_state), (260, 70), eyes_color)
            status_text = "ON" if self.mode == RunMode.STANDALONE else "ON (Stream)"
            FaceApp._draw_text(frame, f"Blink: {status_text}", (400, 70), (0, 255, 0))
        else:
            FaceApp._draw_text(frame, "Blink: OFF", (10, 70), (255, 0, 0))

        FaceApp._draw_text(frame, "Face detection:", (10, 110), (0, 0, 255))
        FaceApp._draw_text(frame, str(detection_results), (260, 110), face_color)
        FaceApp._draw_text(frame, "Distance:", (10, 150), (0, 0, 255))
        FaceApp._draw_text(frame, str(detection_distance), (150, 150), face_color)

        # 顯示執行模式
        if self.mode == RunMode.FASTAPI:
            FaceApp._draw_text(frame, "[FastAPI Mode]", (10, 30), (255, 165, 0))

    def _eyes_preprocessing(
        self, frame: np.ndarray, bounding_eye_left: list, bounding_eye_right: list, threshold_value: int
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Eyes preprocessing."""
        if self.sys_config.debug:
            FaceApp._draw_rectangle(frame, bounding_eye_left)
            FaceApp._draw_rectangle(frame, bounding_eye_right)

        eye_left_roi = frame[
            bounding_eye_left[0][1] : bounding_eye_left[1][1], bounding_eye_left[0][0] : bounding_eye_left[1][0]
        ]
        eye_right_roi = frame[
            bounding_eye_right[0][1] : bounding_eye_right[1][1], bounding_eye_right[0][0] : bounding_eye_right[1][0]
        ]

        # blink detection
        left_eye_gary, right_eye_gary = calculation.Calculation.preprocess_eye_regions(
            eye_left_roi, eye_right_roi, threshold_value
        )
        return left_eye_gary, right_eye_gary

    def _fps_counter(self):
        if time.time() - self.start_time >= 1:
            self.fps = self.fps_count
            self.fps_count = 0
            self.start_time = time.time()

    def toggle_blink_detection(self):
        """Switch blink detection on/off. Use for standalone mode."""
        self.blink_detector.set_enabled(not self.blink_detector.enabled)
        config.logger.info(f"Blink detection toggled to: {self.blink_detector.enabled}")

    # TAG: FastAPI mode methods
    async def _put_frame_async(self, frame: np.ndarray):
        """Put frame into queue (FastAPI mode)"""
        if self.mode != RunMode.FASTAPI or not self.frame_queue:
            return

        try:
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_base64 = base64.b64encode(buffer).decode("utf-8")

            if not self.frame_queue.full():
                await self.frame_queue.put(frame_base64)
        except Exception as e:
            config.logger.error(f"Error putting frame to queue: {e}")

    # TAG: FastAPI mode methods
    async def _put_log_async(self, log_data: dict):
        """Put log data into queue (FastAPI mode)"""
        if self.mode != RunMode.FASTAPI or not self.log_queue:
            return

        try:
            if not self.log_queue.full():
                await self.log_queue.put(log_data)
        except Exception as e:
            config.logger.error(f"Error putting log to queue: {e}")

    def _save_face_image(self, face_roi: np.ndarray, success: bool, person_name: Optional[str] = None) -> str:
        """Save face image to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        status = "success" if success else "failed"

        if self.mode == RunMode.FASTAPI:
            base_dir = Path("captured_faces") / datetime.now().strftime("%Y%m%d")
        else:
            base_dir = Path("captured_faces")

        base_dir.mkdir(parents=True, exist_ok=True)

        if person_name:
            filename = f"{timestamp}_{status}_{person_name}.jpg"
        else:
            filename = f"{timestamp}_{status}.jpg"

        filepath = base_dir / filename
        cv2.imwrite(str(filepath), face_roi)
        config.logger.info(f"Saved face image: {filepath}")

        return str(filepath)

    def _upload_face_image_to_s3(self, face_roi: np.ndarray, detection_results: bool, s3_object_key: str) -> bool:
        """Upload face image to S3."""
        frame_bytes = cv2.imencode(".jpg", face_roi)[1].tobytes()
        upload_status = False

        try:
            upload_status, message = self.minio_client.upload_object(
                bucket_name="face-activity-logs" if detection_results else "face-alarm-logs",
                absolute_path_or_binary=frame_bytes,
                s3_object_key=s3_object_key,
                is_binary=True,
            )
        except Exception as e:
            config.logger.error(f"Error uploading to MinIO S3: {e}")
            return False
        config.logger.info(f"Upload status: {upload_status}")

        return upload_status

    def run(self):
        # If running in FastAPI mode, create an event loop
        loop = None
        if self.mode == RunMode.FASTAPI:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # fps parameters
        self.start_time = time.time()

        # detection parameters
        face_roi: Optional[np.array] = None
        face_in_detection_range: bool = False
        enable_execution_interval: bool = False
        interval_count: int = 0
        detection_results: bool = False
        detection_distance: int = 0
        person_name: Optional[str] = None

        with self.mp_face_detection as face_detection:
            while self.running and self.video_capture.status_alive:
                try:
                    self.fps_count += 1
                    self._fps_counter()

                    if self.video_queue.empty():
                        cv2.waitKey(1) if self.mode == RunMode.STANDALONE else time.sleep(0.001)
                        continue

                    # Handle key events
                    key = cv2.waitKey(1) if self.mode == RunMode.STANDALONE else -1
                    # Press 'b' to toggle blink detection
                    if key == ord("b") or key == ord("B"):
                        self.toggle_blink_detection()

                    frame = cv2.resize(
                        self.video_queue.get(),
                        (self.video_config.image_width, self.video_config.image_height),
                        interpolation=cv2.INTER_AREA,
                    )

                    FaceApp._draw_rectangle(
                        frame,
                        [self.video_config.detection_range_start_point, self.video_config.detection_range_end_point],
                    )

                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(frame_bgr)

                    if results.detections:
                        for detection_mp in results.detections:
                            # face bounding box
                            bounding_box_mp = detection_mp.location_data.relative_bounding_box
                            bounding_box_height = round(bounding_box_mp.height, 2)
                            detection_score = round(detection_mp.score[0], 2)
                            face_bounding_box, center = self.calculation.get_face_boundingbox(bounding_box_mp)

                            if not self.coordinate_detection.face_box_in_roi(
                                center, bounding_box_height, detection_score
                            ):
                                self.blink_detector.reset()
                                face_in_detection_range = False
                                continue

                            self.blink_detector.increment_count()

                            face_roi = frame[
                                face_bounding_box[0][1] : face_bounding_box[1][1],
                                face_bounding_box[0][0] : face_bounding_box[1][0],
                            ]
                            face_in_detection_range = np.all(np.array(face_roi.shape) != 0)

                            # Update brightness for blink detection
                            if (
                                self.blink_detector.enabled
                                and face_in_detection_range
                                and self.blink_detector.should_update_brightness()
                            ):
                                self.blink_detector.update_brightness(
                                    face_roi,
                                    self.reco_config.eyes_detection_brightness_threshold,
                                    self.reco_config.eyes_detection_brightness_value,
                                )

                            if self.sys_config.debug:
                                FaceApp._draw_rectangle(frame, face_bounding_box)

                        # Handles blink detection and facial recognition
                        if self.reco_config.enable and face_in_detection_range:
                            # Blink detection
                            blink_state = False
                            if self.blink_detector.enabled and self.blink_detector.average_brightness != 0:
                                # eyes bounding box
                                bounding_eye_left, bounding_eye_right = self.calculation.get_eyes_boundingbox(
                                    detection_mp, bounding_box_mp.height
                                )
                                left_eye_gary, right_eye_gary = self._eyes_preprocessing(
                                    frame, bounding_eye_left, bounding_eye_right, self.blink_detector.threshold_value
                                )

                                blink_state = self.blink_detector.process_eyes(left_eye_gary, right_eye_gary)

                                if self.mode == RunMode.STANDALONE and self.sys_config.debug:
                                    cv2.imshow("eyes_left", left_eye_gary)
                                    cv2.imshow("eyes_right", right_eye_gary)

                            # Face recognition trigger
                            trigger_recognition = False
                            if self.mode == RunMode.STANDALONE and self.sys_config.debug:
                                # Debug: Press 'r' to trigger recognition
                                trigger_recognition = key == ord("r") or key == ord("R")
                            else:
                                # Production: Use blink detection state
                                if self.blink_detector.enabled:
                                    trigger_recognition = blink_state and not enable_execution_interval
                                else:
                                    trigger_recognition = not enable_execution_interval

                            if trigger_recognition:
                                extraction = threading.Thread(
                                    target=self.predictor.face_prediction,
                                    args=(
                                        face_roi,
                                        self.detection_results_queue,
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

                        # standalone mode: save face features
                        if self.reco_config.set_mode and face_roi is not None:
                            if key == ord("s") or key == ord("S"):
                                face_descriptor, feature_coordinates = self.predictor.feature_extraction(face_roi)
                                name = "User_" + hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
                                predictor.Predictor.save_feature(self.reco_config.face_model, face_descriptor, name)
                                if self.sys_config.debug:
                                    FaceApp._draw_dlib_features(face_roi, feature_coordinates)
                    else:
                        self.blink_detector.reset()
                        face_roi = None

                    # Handle detection results
                    if not self.detection_results_queue.empty():
                        detection_result = self.detection_results_queue.get()
                        detection_distance = round(detection_result[1], 2)
                        detection_results = detection_result[0]
                        person_name = detection_result[2] if len(detection_result) > 2 else "Unknown"

                        # FastAPI mode: save face image and put log
                        if self.mode == RunMode.FASTAPI and frame is not None:
                            if not self.sys_config.debug:
                                # self._save_face_image(frame, detection_results, person_name)
                                s3_object_key = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{person_name}.jpg"
                                upload_status = self._upload_face_image_to_s3(
                                    face_roi, detection_results, s3_object_key
                                )

                            log_data = {
                                "name": person_name,
                                "group": "Unknown",
                                "s3_object_key": s3_object_key if upload_status else None,
                                "detection_results": detection_results,
                                "timestamp": datetime.now().isoformat(),
                            }

                            # Send log data
                            if loop:
                                loop.run_until_complete(self._put_log_async(log_data))

                    # Draw results
                    self._draw_result_information(
                        frame, detection_results, self.blink_detector.blink_state, detection_distance
                    )

                    if self.sys_config.debug:
                        FaceApp._draw_text(frame, "FPS: " + str(self.fps), (10, 30), (0, 0, 255))

                    # Handle frame display
                    if self.mode == RunMode.STANDALONE:
                        # Standalone mode: Show video window
                        cv2.imshow("video_out", frame)
                        if key == ord("q") or key == ord("Q"):
                            break
                    else:
                        # FastAPI mode: put frame into queue
                        if self.fps_count % 3 == 0 and loop:
                            loop.run_until_complete(self._put_frame_async(frame))

                    if self.mode == RunMode.STANDALONE and self.sys_config.debug:
                        cv2.imshow("video_out", frame)
                        debug_key = cv2.waitKey(1)
                        if debug_key == ord("q") or debug_key == ord("Q"):
                            break
                        elif debug_key == ord("b") or debug_key == ord("B"):
                            self.toggle_blink_detection()

                except Exception as e:
                    config.logger.debug(f"Error in main loop: {e}")
                    if self.mode == RunMode.STANDALONE:
                        time.sleep(1)
                        break
                    else:
                        config.logger.error("Error in main loop, continuing...")
                        time.sleep(0.1)

        # 清理資源
        cv2.destroyAllWindows()
        if loop:
            print("Stopping event loop...")
            loop.close()
        self.stop()


if __name__ == "__main__":
    app = FaceApp(mode=RunMode.STANDALONE)
    app.run()
