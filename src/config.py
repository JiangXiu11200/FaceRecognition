import csv
import datetime
import json
import logging
import logging.config
import numpy as np
import os

SETTING_DIRECTORY = "../setting.json"
FACE_MODEL = "./model.csv"
LOGS_DIRECTORY = "./logs" 
TODAY_LOGS = LOGS_DIRECTORY + "/" + str(datetime.date.today())

face_model = []

with open(SETTING_DIRECTORY) as f:
    data = json.load(f)
    height = data["sys_config"]["height"]
    width = data["sys_config"]["width"]
    debug = data["debug"]
    mode = data["mode"]

os.makedirs(TODAY_LOGS, exist_ok=True)
logging.config.fileConfig("./logger_config.conf", defaults={'logfilename': TODAY_LOGS + "/system.log"})
logger = logging.getLogger('root')

if mode == 1:
    with open(FACE_MODEL) as model:
        rows = csv.reader(model)
        for row in rows:
            face_model.append(np.array(row, dtype=float))
    face_model = np.array(face_model)