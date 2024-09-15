import csv
import json
import os

import dlib
import numpy as np

import package.config as config


class VideoConfig:
    def __init__(self, rtsp: str, image_height: int, image_width: int, detection_range_start_point: list, detection_range_end_point: list):
        __slots__ = ["rtsp", "image_height", "image_width", "detection_range_start_point", "detection_range_end_point"]
        self.rtsp = rtsp
        self.image_height = image_height
        self.image_width = image_width
        self.detection_range_start_point = detection_range_start_point
        self.detection_range_end_point = detection_range_end_point


class SystemConfig:
    def __init__(self, debug: bool, mode: str, logs_path: str):
        __slots__ = ["debug", "mode", "logs_path"]
        self.debug = debug
        self.mode = mode
        self.logs_path = logs_path


class RecoConfig:
    def __init__(
        self,
        enable: bool,
        set_mode: bool,
        dlib_predictor: str,
        dlib_recognition_model: str,
        face_model: str,
        minimum_bounding_box_height: int,
        minimum_face_detection_score: float,
        eyes_detection_brightness_threshold: int,
        eyes_detection_brightness_value: list,
        sensitivity: float,
        consecutive_prediction_intervals: int,
    ):
        __slots__ = [
            "enable",
            "set_mode",
            "dlib_predictor",
            "dlib_recognition_model",
            "face_model",
            "minimum_bounding_box_height",
            "minimum_face_detection_score",
            "face_features",
            "eyes_detection_brightness_threshold",
            "eyes_detection_brightness_value",
            "sensitivity",
            "consecutive_prediction_intervals",
        ]
        self.enable = enable
        self.set_mode = set_mode
        self.dlib_predictor = dlib.shape_predictor(dlib_predictor)
        self.dlib_recognition_model = dlib.face_recognition_model_v1(dlib_recognition_model)
        self.face_model = face_model
        self.minimum_bounding_box_height = minimum_bounding_box_height
        self.minimum_face_detection_score = minimum_face_detection_score
        self.eyes_detection_brightness_threshold = eyes_detection_brightness_threshold
        self.eyes_detection_brightness_value = eyes_detection_brightness_value
        self.sensitivity = sensitivity
        self.consecutive_prediction_intervals = consecutive_prediction_intervals
        self.registered_face_descriptor: np.ndarray = None
        self.load_face_features()

    def load_face_features(self):
        face_features = []
        if os.path.isfile(self.face_model):
            with open(self.face_model) as model:
                rows = csv.reader(model)
                for row in rows:
                    face_features.append(np.array(row, dtype=float))
        else:
            with open(self.face_model, mode="a"):
                pass
        self.registered_face_descriptor = np.array(face_features)


class Settings:
    def __init__(self):
        self.video_config: VideoConfig = None
        self.system_config: SystemConfig = None
        self.reco_config: RecoConfig = None

    def load_setting(self):
        with open(config.SETTING_DIRECTORY) as f:
            data = json.load(f)
            self.updata_setting(data["video_config"], data["sys_config"], data["reco_config"])
        if self.system_config.debug == True:
            self.reco_config.consecutive_prediction_intervals = 9999

    def updata_setting(self, video_config, system_config, reco_config):
        self.video_config = VideoConfig(**video_config)
        self.system_config = SystemConfig(**system_config)
        self.reco_config = RecoConfig(**reco_config)
