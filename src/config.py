import json
import os

SETTING_DIRECTORY = "../setting.json"

with open(SETTING_DIRECTORY) as f:
    data = json.load(f)
    height = data["sys_config"]["height"]
    width = data["sys_config"]["width"]
    debug = data["debug"]