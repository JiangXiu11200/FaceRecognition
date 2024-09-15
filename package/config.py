import datetime
import logging
import logging.config
import os

# system config
SETTING_DIRECTORY = "setting.json"

# logger config
LOGGER_CONFIG = "logger_config.conf"
LOGGER_DIRECTORY = "logs" + "/" + str(datetime.date.today())
os.makedirs(LOGGER_DIRECTORY, exist_ok=True)
LOGGER_PATH = LOGGER_DIRECTORY + "/system.log"
logging.config.fileConfig(LOGGER_CONFIG, defaults={"logfilename": LOGGER_PATH})
logger = logging.getLogger("root")
