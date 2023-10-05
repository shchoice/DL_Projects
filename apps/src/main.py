import logging

import uvicorn
from fastapi import FastAPI

from apps.src.config import constants
from apps.src.config.logging_config import LOGGING_CONFIG
from utils.yaml.load import load_yaml_config
from router import classifier_router


app = FastAPI()
app.include_router(classifier_router)
@app.on_event("startup")
async def startup_event():
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(constants.LOGGING_LEVEL)
    logger.info("Application startup complete!!")


if __name__ == '__main__':
    uvicorn_config = load_yaml_config(constants.CONFIG_UVICORN_YAML_FILE_NAME)
    uvicorn.run(
        app='main:app',
        host=uvicorn_config['host'],
        port=uvicorn_config['port'],
        workers=uvicorn_config['workers'],
        log_level=uvicorn_config['log_level'],
        timeout_keep_alive=uvicorn_config['timeout_keep_alive']
    )
