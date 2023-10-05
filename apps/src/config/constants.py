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
