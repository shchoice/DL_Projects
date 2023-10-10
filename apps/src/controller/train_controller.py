import logging
import sys
from typing import Dict, Any

from fastapi import Response, status
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter

from apps.src.config import constants
from apps.src.exception.train_exception import TrainException
from apps.src.schemas.train_config import TrainConfig
from apps.src.service.train_service import TrainService
from apps.src.utils.log.log_message import LogMessage
from apps.src.utils.yaml.load import load_train_config

router = InferringRouter()


@cbv(router)
class TrainController:
    def __init__(self):
        self.logger = logging.getLogger(constants.LOGGER_INFO_NAME)
        self.log_message = LogMessage()

    @router.post('/train')
    def train_controller(self, train_config: TrainConfig, response: Response) -> Dict[str, Any]:
        try:
            train_config = load_train_config(
                yaml_file=constants.CONFIG_TRAIN_YAML_FILE_NAME,
                schema=train_config
            )

            train_service = TrainService(train_config)
            train_service.run_classifier()

            response.status_code = status.HTTP_200_OK
            self.logger.info("Train controller execution was successful")

            return {"Message": "Success"}
        except TrainException as te:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.logger.error(self.log_message.make_log_message(
                    line_no=self.log_message.get_line_number(exc_traceback),
                    stack_trace=self.log_message.stack_trace(exc_type, exc_value, exc_traceback)
                )
            )
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

            return {"Error": str(te)}
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.logger.error(self.log_message.make_log_message(
                    line_no=self.log_message.get_line_number(exc_traceback),
                    stack_trace=self.log_message.stack_trace(exc_type, exc_value, exc_traceback)
                )
            )
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

            return {"Error": "An unexpected error occurred." + str(e)}
