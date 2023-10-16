import logging

from fastapi import Response, status
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter

from apps.src.config import constants
from apps.src.utils.log.log_message import LogMessage

router = InferringRouter()


@cbv(router)
class HealthController:
    def __init__(self):
        self.logger = logging.getLogger(constants.LOGGER_INFO_NAME)
        self.log_message = LogMessage()

    @router.get('/health')
    def health_check_controller(self, response: Response):
        try:
            response.status_code = status.HTTP_200_OK
            return {"Message": "Health check : All systems are operational"}
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.logger.error(self.log_message.make_log_message(
                line_no=self.log_message.get_line_number(exc_traceback),
                stack_trace=self.log_message.stack_trace(exc_type, exc_value, exc_traceback)
            )
            )
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

            return {"Error": "An unexpected error occurred." + str(e)}
