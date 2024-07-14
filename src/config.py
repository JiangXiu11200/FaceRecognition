import csv
import datetime
import json
import logging
import logging.config
import os

import dlib
import numpy as np

SETTING_DIRECTORY = "setting.json"
LOGGER_CONFIG = "logger_config.conf"
FACE_RECOGNITION_MODE = False
SETTING_MODE = False

with open(SETTING_DIRECTORY) as f:
    data = json.load(f)
    image_height = data["video_config"]["image_height"]
    image_width = data["video_config"]["image_width"]
    detection_range_start_point = data["video_config"]["detection_range_start_point"]
    detection_range_end_point = data["video_config"]["detection_range_end_point"]
    minimum_bounding_box_height = data["reco_config"]["minimum_bounding_box_height"]
    minimum_face_detection_score = data["reco_config"]["minimum_face_detection_score"]
    dlib_predictor = dlib.shape_predictor(data["dlib_predictor"])
    dlib_recognition_model = dlib.face_recognition_model_v1(data["dlib_recognition_model"])
    face_model = data["face_model"]
    debug = data["sys_config"]["debug"]
    mode = data["sys_config"]["mode"]
    logs_path = data["logs_path"]

if mode == "recognition":
    FACE_RECOGNITION_MODE = True
if mode == "setting":
    SETTING_MODE = True

TODAY_LOGS = logs_path + "/" + str(datetime.date.today())
os.makedirs(TODAY_LOGS, exist_ok=True)
logging.config.fileConfig(LOGGER_CONFIG, defaults={'logfilename': TODAY_LOGS + "/system.log"})
logger = logging.getLogger('root')

if FACE_RECOGNITION_MODE:
    face_features = []
    with open(face_model) as model:
        rows = csv.reader(model)
        for row in rows:
            face_features.append(np.array(row, dtype=float))
    face_features = np.array(face_features)