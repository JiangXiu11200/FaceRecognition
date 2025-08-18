import csv
import os
from dataclasses import dataclass, field

import dlib
import numpy as np
from dynaconf import Dynaconf

import package.config as config


@dataclass
class VideoConfig:
    __solts__ = ["rtsp", "image_height", "image_width", "detection_range_start_point", "detection_range_end_point"]
    rtsp: str
    image_height: int
    image_width: int
    detection_range_start_point: list
    detection_range_end_point: list


@dataclass
class SystemConfig:
    __slots__ = ["debug", "logs_path"]
    debug: bool
    logs_path: str


@dataclass
class RecoConfig:
    enable: bool
    set_mode: bool
    enable_blink_detection: bool
    dlib_predictor: str
    dlib_recognition_model: str
    face_model: str
    minimum_bounding_box_height: int
    minimum_face_detection_score: float
    eyes_detection_brightness_threshold: int
    eyes_detection_brightness_value: list
    sensitivity: float
    consecutive_prediction_intervals: int
    registered_face_descriptor: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        # Initialize dlib models
        self.dlib_predictor = dlib.shape_predictor(self.dlib_predictor)
        self.dlib_recognition_model = dlib.face_recognition_model_v1(self.dlib_recognition_model)
        # Load face features
        self.load_face_features()

    def load_face_features(self):
        face_features = []
        if os.path.isfile(self.face_model):
            with open(self.face_model) as model:
                rows = csv.reader(model)
                for row in rows:
                    face_features.append(np.array(row, dtype=float))
        else:
            # Create the directory if it does not exist
            directory_path = os.path.dirname(self.face_model)
            os.makedirs(directory_path, exist_ok=True)
            with open(self.face_model, mode="a"):
                pass
        self.registered_face_descriptor = np.array(face_features)


class Settings:
    def __init__(self):
        self.video_config: VideoConfig = None
        self.system_config: SystemConfig = None
        self.reco_config: RecoConfig = None

    def load_setting(self):
        settings = Dynaconf(settings_files=[config.SETTING_DIRECTORY])
        self.updata_setting(settings.video_config, settings.sys_config, settings.reco_config)
        # if self.system_config.debug == True:
        #     self.reco_config.consecutive_prediction_intervals = 9999

    def updata_setting(self, video_config, system_config, reco_config):
        self.video_config = VideoConfig(**video_config)
        self.system_config = SystemConfig(**system_config)
        self.reco_config = RecoConfig(**reco_config)
