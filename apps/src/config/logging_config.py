from pathlib import Path
import sys

import os
import re
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler

from apps.src.config import constants
from apps.src.utils.yaml.load import load_yaml_config


class DqTimedRotatingFileHandler(TimedRotatingFileHandler):
    def __init__(self, filename_template, *args, **kwargs):
        self.filename_template = filename_template
        self.init = True
        super().__init__(self.generate_filename(), *args, **kwargs)

        if self.when == 'S':
            self.interval = 1  # one second
            self.suffix = "%Y-%m-%d_%H-%M-%S"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(\.\w+)?$"
        elif self.when == 'M':
            self.interval = 60  # one minute
            self.suffix = "%Y-%m-%d_%H-%M"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}(\.\w+)?$"
        elif self.when == 'H':
            self.interval = 60 * 60  # one hour
            self.suffix = "%Y-%m-%d_%H"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}_\d{2}(\.\w+)?$"
        elif self.when == 'D' or self.when == 'MIDNIGHT':
            self.interval = 60 * 60 * 24  # one day
            self.suffix = "%Y-%m-%d"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}(\.\w+)?$"
        elif self.when.startswith('W'):
            self.interval = 60 * 60 * 24 * 7  # one week
            if len(self.when) != 2:
                raise ValueError("You must specify a day for weekly rollover from 0 to 6 (0 is Monday): %s" % self.when)
            if self.when[1] < '0' or self.when[1] > '6':
                raise ValueError("Invalid day specified for weekly rollover: %s" % self.when)
            self.dayOfWeek = int(self.when[1])
            self.suffix = "%Y-%m-%d"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}(\.\w+)?$"
        else:
            raise ValueError("Invalid rollover interval specified: %s" % self.when)
        self.extMatch = re.compile(self.extMatch, re.ASCII)

    def generate_filename(self):
        if self.init is True:
            import os
            dir_name, baseName = os.path.split(self.filename_template)
            file_name = baseName.split('-')[0] + "-" + str(os.getpid())
            self.init = False
            return dir_name + os.sep + file_name + constants.LOGGING_EXTENSION_NAME

        one_time_ago = self.get_datetime()
        previous_date = one_time_ago.strftime(self.suffix)

        return self.filename_template.format(date=previous_date)

    def get_datetime(self):
        if self.when == 'S':
            return datetime.now() - timedelta(seconds=1)
        elif self.when == 'M':
            return datetime.now() - timedelta(minutes=1)
        elif self.when == 'H':
            return datetime.now() - timedelta(hours=1)
        elif self.when == 'D' or self.when == 'MIDNIGHT':
            return datetime.now() - timedelta(days=1)
        elif self.when.startswith('W'):
            return datetime.now() - timedelta(weeks=1)

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None

        new_filename = self.generate_filename()
        os.rename(self.baseFilename, new_filename)

        self.mode = "a"
        self.stream = self._open()

        if self.backupCount > 0:
            files_to_delete = self.getFilesToDelete()
            for s in files_to_delete[-self.backupCount:]:
                os.remove(s)

    def getFilesToDelete(self):
        """
        Determine the files to delete when rolling over.

        More specific than the earlier method, which just used glob.glob().
        """
        dir_name, base_name = os.path.split(self.baseFilename)
        base_name = base_name[:-4]
        file_names = os.listdir(dir_name)
        result = []
        prefix = base_name + "-"
        plen = len(prefix)
        for file_name in file_names:
            if file_name[:plen] == prefix:
                suffix = file_name[plen:]
                if self.extMatch.match(suffix):
                    result.append(os.path.join(dir_name, file_name))
        if len(result) < self.backupCount:
            result = []
        else:
            result.sort()
            result = result[:len(result) - self.backupCount]
        return result


VOLUME_BASE_DIR = str(Path(__file__).resolve().parent.parent)
VOLUME_ACCESS_LOG_PATH = VOLUME_BASE_DIR + '/volumes/logs/access.log'
VOLUME_INFO_LOG_PATH = VOLUME_BASE_DIR + '/volumes/logs/info.log'
VOLUME_ERROR_LOG_PATH = VOLUME_BASE_DIR + '/volumes/logs/error.log'

log_config = load_yaml_config(constants.CONFIG_LOGGING_YAML_FILE_NAME)

LOGGING_CONFIG = {
    "version": 1,
    "loggers": {
        "root": {
            "level": "INFO",
            "handlers": ["console", "info_file"],
            "propagate": 1
        },
        "uvicorn": {
            "level": "INFO",
            "handlers": ["console", "info_file"],
            "propagate": 0,
            "qualname": "uvicorn",
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
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "console",
            "stream": "ext://sys.stdout",
        },
        "access_file": {
            "class": "logging_conf.DqTimedRotatingFileHandler",
            "formatter": "access",
            "filename_template": VOLUME_ACCESS_LOG_PATH[:-4] + "-{date}.log",
            "when": log_config['rotate']['when'],
            "backupCount": log_config['rotate']['backupCount'],
        },
        "info_file": {
            "class": "logging_conf.DqTimedRotatingFileHandler",
            "formatter": "generic",
            "filename_template": VOLUME_INFO_LOG_PATH[:-4] + "-{date}.log",
            "when": log_config['rotate']['when'],
            "backupCount": log_config['rotate']['backupCount'],
        },
        "error_file": {
            "class": "logging_conf.DqTimedRotatingFileHandler",
            "formatter": "error",
            "filename_template": VOLUME_ERROR_LOG_PATH[:-4] + "-{date}.log",
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

