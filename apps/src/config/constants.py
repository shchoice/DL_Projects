import os
from typing_extensions import Final

# CONFIG
CONFIG_PATH_NAME: Final = 'config'
CONFIG_UVICORN_YAML_FILE_NAME: Final = os.path.join(CONFIG_PATH_NAME, 'asgi.yaml')
CONFIG_LOGGING_YAML_FILE_NAME: Final = CONFIG_PATH_NAME + os.sep + 'logging.yaml'
CONFIG_DATA_YAML_FILE_NAME: Final = os.path.join(CONFIG_PATH_NAME, 'data.yaml')

# LOGGER
LOGGER_INFO_NAME: Final = 'uvicorn.info'
LOGGER_ERROR_NAME: Final = 'uvicorn.error'
LOGGING_LEVEL: Final = 'uvicorn.info'
LOGGING_EXTENSION_NAME: Final = '.log'

DATA_COLUMN_SEP: Final = '\t'
DATA_FILE_ENCODING: Final = 'UTF-8'
DATA_PATH_NAME: Final = 'volumes/data'
DATA_RAW_PATH_NAME: Final = 'raw'
DATA_TRAIN_PATH_NAME: Final = 'train'
DATA_VALID_PATH_NAME: Final = 'valid'
DATA_TRAIN_VALID_PATH_NAME: Final = 'train_valid'
DATA_TEST_PATH_NAME: Final = 'test'
