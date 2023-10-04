import os
from typing_extensions import Final

# CONFIG
CONFIG_PATH_NAME: Final = 'config'
CONFIG_UVICORN_YAML_FILE_NAME: Final = os.path.join(CONFIG_PATH_NAME, 'asgi.yaml')
CONFIG_LOGGING_YAML_FILE_NAME: Final = CONFIG_PATH_NAME + os.sep + 'logging.yaml'
CONFIG_DATA_YAML_FILE_NAME: Final = os.path.join(CONFIG_PATH_NAME, 'data.yaml')
