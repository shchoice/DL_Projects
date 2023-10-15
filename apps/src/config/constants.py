import os
from typing_extensions import Final

# CONFIG
CONFIG_PATH_NAME: Final = 'config'
CONFIG_UVICORN_YAML_FILE_NAME: Final = os.path.join(CONFIG_PATH_NAME, 'asgi.yaml')
CONFIG_LOGGING_YAML_FILE_NAME: Final = os.path.join(CONFIG_PATH_NAME, 'logging.yaml')
CONFIG_DATA_YAML_FILE_NAME: Final = os.path.join(CONFIG_PATH_NAME, 'data.yaml')
CONFIG_TRAIN_YAML_FILE_NAME: Final = os.path.join(CONFIG_PATH_NAME, 'train.yaml')

# LOGGER
LOGGER_INFO_NAME: Final = 'uvicorn.info'
LOGGER_ERROR_NAME: Final = 'uvicorn.error'
LOGGING_LEVEL: Final = 'uvicorn.info'
LOGGING_EXTENSION_NAME: Final = '.log'

# DATA
DATA_COLUMN_SEP: Final = '\t'
DATA_FILE_ENCODING: Final = 'UTF-8'
DATA_PATH_NAME: Final = 'apps/volumes/data'
DATA_RAW_PATH_NAME: Final = 'raw'
DATA_TRAIN_PATH_NAME: Final = 'train'
DATA_VALID_PATH_NAME: Final = 'valid'
DATA_TRAIN_VALID_PATH_NAME: Final = 'train_valid'
DATA_TEST_PATH_NAME: Final = 'test'

# MODEL
MODEL_PATH_NAME: Final = 'apps/volumes/models'
MODEL_KOBERT_CARD_NAME: Final = 'skt/kobert-base-v1'
MODEL_KOBERT_FINAL: Final = 'final'
MODEL_CONFIG_YAML_FILE_NAME: Final = 'config.yaml'
MODEL_CONFIG_PATH_NAME: Final = 'apps/volumes/config'

# OUTPUT
OUTPUT_PATH_NAME: Final = 'apps/volumes/output'

# LABEL ENCODER
LABEL_ENCODER_NAME: Final = 'label_encoder.pkl'
