import logging
import sys
import time
from typing import Dict, Any

from fastapi import Response, status
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter

from apps.src.config import constants
from apps.src.exception.model_exchange_exception import ModelExchangeException
from apps.src.exception.predict_exception import PredictException
from apps.src.modules.common.config_manager import ConfigManager
from apps.src.schemas.predict_schema import PredictSchema
from apps.src.schemas.reload_schema import ReloadSchema
from apps.src.service.prediction_service import PredictionService
from apps.src.utils.json.make_predict_json import set_hits_json, set_response_json
from apps.src.utils.log.log_message import LogMessage
from apps.src.utils.yaml.load import load_reload_model_config

router = InferringRouter()


@cbv(router)
class PredictionController:
    def __init__(self):
        self.logger = logging.getLogger(constants.LOGGER_INFO_NAME)
        self.log_message = LogMessage()

    @router.post('/predict')
    def prediction_controller(self, predict_schema: PredictSchema, response: Response) -> Dict[str, Any]:
        try:
            start_time = time.perf_counter()
            predict_config = ConfigManager.configure(config_type="Prediction", schema=predict_schema)

            predict_service = PredictionService(predict_config)
            top_k_decoded_labels, top_k_values = predict_service.run_predict()

            exec_time = time.perf_counter() - start_time
            hits_json = set_hits_json(predict_config['documents'], top_k_decoded_labels, top_k_values)
            response_json = set_response_json(hits_json, predict_config, exec_time)

            response.status_code = status.HTTP_200_OK
            self.logger.info("Prediction service execution was successful")

            return {"message": "success", "json": response_json}
        except PredictException as pe:
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

    @router.post('/model/exchange')
    def model_exchange(self, reload_config: ReloadSchema, response: Response) -> Dict[str, Any]:
        try:
            reload_config = load_reload_model_config(schema=reload_config)

            prediction_service = PredictionService(reload_config)
            prediction_service.run_model_exchange()

            response.status_code = status.HTTP_200_OK
            self.logger.info("Model exchange execution was successful")

            return {"message": "success"}
        except ModelExchangeException as mee:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.logger.error(self.log_message.make_log_message(
                    line_no=self.log_message.get_line_number(exc_traceback),
                    stack_trace=self.log_message.stack_trace(exc_type, exc_value, exc_traceback)
                )
            )
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

            return {"Error": str(mee)}
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.logger.error(self.log_message.make_log_message(
                    line_no=self.log_message.get_line_number(exc_traceback),
                    stack_trace=self.log_message.stack_trace(exc_type, exc_value, exc_traceback)
                )
            )
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

            return {"Error": "An unexpected error occurred." + str(e)}
