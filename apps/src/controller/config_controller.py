import logging
import sys

from fastapi import Response, status
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter

from apps.src.config import constants
from apps.src.modules.common.config_manager import ConfigManager
from apps.src.schemas.update_schema import UpdateSchema
from apps.src.utils.log.log_message import LogMessage

router = InferringRouter()


@cbv(router)
class ConfigController:
    def __init__(self):
        self.logger = logging.getLogger(constants.LOGGER_INFO_NAME)
        self.log_message = LogMessage()

    @router.patch('/config/update')
    def config_update_contoller(self, update_schema: UpdateSchema, response: Response):
        try:
            config = ConfigManager.get_config_instance(
                update_schema.model_type, update_schema.base_dir, update_schema.text_dataset
            )
            update_schema = self.set_key_mapping(dict(update_schema))

            for key, value in update_schema.items():
                if value is not None:
                    config.set_model_config([key], value)

            config.save_model_config()

            response.status_code = status.HTTP_200_OK
            return {"message": "Configuration updated successfully"}
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.logger.error(self.log_message.make_log_message(
                line_no=self.log_message.get_line_number(exc_traceback),
                stack_trace=self.log_message.stack_trace(exc_type, exc_value, exc_traceback)
            ))
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

            return {"Error": "An unexpected error occurred." + str(e)}

    def set_key_mapping(self, update_schema):
        new_update_schema = {}
        model_type = update_schema.get('model_type')
        key_path_mapping = {
            'model_type': ['model_type'],
            'text_dataset': ['text_dataset'],
            'base_dir': ['base_dir'],
            'gpu_id': ['gpu_id'],
            'load_trained_model': ['load_trained_model'],
            'load_model_name': ['load_model_name'],
            'filename_extension': ['filename_extension'],

            'padding': ['tokenizer', 'padding'],

            'delimiter': ['dataloader', 'delimiter'],
            'column_names': ['dataloader', 'column_names'],

            'per_device_train_batch_size': ['trainer_args', 'per_device_train_batch_size'],
            'per_device_eval_batch_size': ['trainer_args', 'per_device_eval_batch_size'],
            'num_train_epochs': ['trainer_args', 'num_train_epochs'],
            'evaluation_strategy': ['trainer_args', 'evaluation_strategy'],
            'steps': ['trainer_args', 'steps'],
            'learning_rate': ['trainer_args', 'learning_rate'],
            'optim': ['trainer_args', 'optim'],
            'warmup_ratio': ['trainer_args', 'warmup_ratio'],
            'warmup_steps': ['trainer_args', 'warmup_steps'],
            'lr_scheduler_type': ['trainer_args', 'lr_scheduler_type'],
            'output_dir': ['trainer_args', 'output_dir'],
            'logging_dir': ['trainer_args', 'logging_dir'],
            'weight_decay': ['trainer_args', 'weight_decay'],
            'early_stopping_patience': ['trainer_args', 'early_stopping_patience'],

            'max_length': [model_type, 'max_length'],
        }

        for key, value in update_schema.items():
            key_path = key_path_mapping.get(key)
            if key_path:
                self.set_nested_dict(new_update_schema, key_path, value)

        return new_update_schema

    @staticmethod
    def set_nested_dict(data_dict, key_path, value):
        for key in key_path[:-1]:
            data_dict = data_dict.setdefault(key, {})
        data_dict[key_path[-1]] = value

