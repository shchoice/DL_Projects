import sys
from datetime import datetime
from pathlib import Path

from apps.src.config import constants
from apps.src.utils.yaml.load import load_yaml_config


CODE_BASE_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, CODE_BASE_DIR + '/utils/log')

VOLUME_BASE_DIR = str(Path(__file__).resolve().parent.parent.parent)
VOLUME_ACCESS_LOG_PATH = VOLUME_BASE_DIR + '/volumes/logs/access' + constants.LOGGING_EXTENSION_NAME
VOLUME_INFO_LOG_PATH = VOLUME_BASE_DIR + '/volumes/logs/info' + constants.LOGGING_EXTENSION_NAME
VOLUME_ERROR_LOG_PATH = VOLUME_BASE_DIR + '/volumes/logs/error' + constants.LOGGING_EXTENSION_NAME

log_config = load_yaml_config(constants.CONFIG_LOGGING_YAML_FILE_NAME)

LOGGING_CONFIG = {
    "version": 1,
    "loggers": {
        "root": {
            "level": "INFO",
            "handlers": ["console", "info_file"],
            "propagate": 1
        },
        "uvicorn.info": {
            "level": "INFO",
            "handlers": ["console", "info_file"],
            "propagate": 0,
            "qualname": "uvicorn.info",
        },
        "uvicorn.access": {
            "level": "INFO",
            "handlers": ["console", "access_file"],
            "propagate": 0,
            "qualname": "uvicorn.access",
        },
        "uvicorn.error": {
            "level": "ERROR",
            "handlers": ["console", "error_file"],
            "propagate": 1,
            "qualname": "uvicorn.error",
        },
        "debug": {
            "level": "DEBUG",
            "handlers": ["console", "info_file"],
            "propagate": 0
        },
        "transformers": {
            "level": "INFO",
            "handlers": ["console", "info_file"],
            "propagate": 0
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "console",
            "stream": "ext://sys.stdout",
        },
        "access_file": {
            "class": "rotating_file_handler.CustomTimedRotatingFileHandler",
            "formatter": "access",
            "filename_template": VOLUME_ACCESS_LOG_PATH[:-4] + datetime.now().strftime('-%Y%m%d') + constants.LOGGING_EXTENSION_NAME,
            "when": log_config['rotate']['when'],
            "backupCount": log_config['rotate']['backupCount'],
        },
        "info_file": {
            "class": "rotating_file_handler.CustomTimedRotatingFileHandler",
            "formatter": "generic",
            "filename_template": VOLUME_INFO_LOG_PATH[:-4] + datetime.now().strftime('-%Y%m%d') + constants.LOGGING_EXTENSION_NAME,
            "when": log_config['rotate']['when'],
            "backupCount": log_config['rotate']['backupCount'],
        },
        "error_file": {
            "class": "rotating_file_handler.CustomTimedRotatingFileHandler",
            "formatter": "error",
            "filename_template": VOLUME_ERROR_LOG_PATH[:-4] + datetime.now().strftime('-%Y%m%d') + constants.LOGGING_EXTENSION_NAME,
            "when": log_config['rotate']['when'],
            "backupCount": log_config['rotate']['backupCount'],
        },
    },
    "formatters": {
        "console": {
            "format": "%(levelname)s: %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
            "class": "uvicorn.logging.ColourizedFormatter",
            "use_colors": True,
        },
        "access": {
            "format": "{asctime} - {message}",
            "style": "{",
            "class": "uvicorn.logging.ColourizedFormatter",
            "use_colors": True,
        },
        "generic": {
            "format": "[%(asctime)s] %(levelname)s - %(name)s on %(module)s:%(lineno)s [%(thread)d] - %(message)s",
            "class": "logging.Formatter",
        },
        "error": {
            "format": "[%(asctime)s] %(levelname)s - %(name)s on %(module)s:%(lineno)s %(funcName)s() [%(thread)d] - %(message)s",
            "class": "logging.Formatter",
        },
    },
}

