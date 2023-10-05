import logging
import sys

from fastapi import Response, status
from typing import Dict, Any

from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter

from apps.src.config import constants
from apps.src.exception.preprocess_exception import PreprocessException
from apps.src.schemas.preprocess_config import PreprocessConfig
from apps.src.utils.log.log_message import LogMessage
from apps.src.utils.yaml.load import load_data_config


router = InferringRouter()

@cbv(router)
class PreprocessController:
    def __init__(self):
        self.logger = logging.getLogger(constants.LOGGER_INFO_NAME)
        self.log_message = LogMessage()

    @router.post('/preprocess')
    def preprocess(self, preprocess_config: PreprocessConfig, response: Response) -> Dict[str, Any]:
        try:
            data_config = load_data_config(
                yaml_file=constants.CONFIG_DATA_YAML_FILE_NAME,
                schema=preprocess_config
            )

            print(data_config)

            response.status_code = status.HTTP_200_OK
            self.logger.info("Preprocessing Succeed")
            return {"Message": "Success"}
        except PreprocessException as pe:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.logger.error(self.log_message.make_log_message(
                    line_no=self.log_message.get_line_number(exc_traceback),
                    stack_trace=self.log_message.stack_trace(exc_type, exc_value, exc_traceback)
                )
            )
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            return {"Error": str(pe)}
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.logger.error(self.log_message.make_log_message(
                    line_no=self.log_message.get_line_number(exc_traceback),
                    stack_trace=self.log_message.stack_trace(exc_type, exc_value, exc_traceback)
                )
            )
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            return {"Error": "An unexpected error occurred." + str(e)}
