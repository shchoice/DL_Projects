import os
from typing import Any, List

import yaml

from apps.src.config import constants


class BaseConfig:
    _default_config = {}

    def __init__(self, model_type: str, base_dir: str, text_dataset: str):
        self.update(self._default_config)
        self._initialize_config(model_type, base_dir, text_dataset)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def update(self, other_dict):
        for key, value in other_dict.items():
            self[key] = value

    def _initialize_config(self, model_type: str, base_dir: str, text_dataset: str):
        self.model_type = model_type
        self.base_dir = base_dir
        self.text_dataset = text_dataset

        config_path = self._get_config_path()

        # 첫번째 model_type, text_dataset 사용 시에는 config.yaml 파일에서 읽어오지 않음(존재X), 2번째 부터 참조
        if os.path.isfile(config_path):
            self._load_config_from_file(config_path)

    def _get_config_path(self) -> str:
        return os.path.join(self.base_dir, constants.MODEL_CONFIG_PATH_NAME,
                            self.text_dataset, self.model_type,
                            constants.MODEL_CONFIG_YAML_FILE_NAME)

    def _load_config_from_file(self, config_path: str):
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                self.update(loaded_config)

    def set_model_config(self, key_path: List[str], value: Any):
        temp = self
        for key in key_path[:-1]:
            if not hasattr(temp, key):
                setattr(temp, key, {})
            temp = getattr(temp, key)
        if not hasattr(temp, key_path[-1]):
            raise KeyError(f"Key '{key_path[-1]}' not found in the configuration.")
        setattr(temp, key_path[-1], value)

    def update_and_save_config(self, train_config):
        if self.model_type == 'KoBERT':
            updated_config = self.get_updated_kobert_config(train_config)
            self.update(updated_config)
            self.save_model_config()


    def get_updated_kobert_config(self, train_config):
        return {
            'model_type': train_config['model_type'],
            'text_dataset': train_config['text_dataset'],
            'base_dir': train_config['base_dir'],
            'gpu_id': train_config['gpu_id'],
            'load_trained_model': train_config['load_trained_model'],
            'load_model_name': train_config['load_model_name'],
            'filename_extension': train_config['filename_extension'],

            'tokenizer': {
                'padding': train_config['tokenizer']['padding'],
            },

            'dataloader': {
                'delimiter': train_config['dataloader']['delimiter'],
                'column_names': train_config['dataloader']['column_names'],
            },

            'trainer_args': {
                'per_device_train_batch_size': train_config['trainer_args']['per_device_train_batch_size'],
                'per_device_eval_batch_size': train_config['trainer_args']['per_device_eval_batch_size'],
                'num_train_epochs': train_config['trainer_args']['num_train_epochs'],
                'evaluation_strategy': train_config['trainer_args']['evaluation_strategy'],
                'steps': train_config['trainer_args']['steps'],
                'learning_rate': train_config['trainer_args']['learning_rate'],
                'optim': train_config['trainer_args']['optim'],
                'warmup_ratio': train_config['trainer_args']['warmup_ratio'],
                'warmup_steps': train_config['trainer_args']['warmup_steps'],
                'lr_scheduler_type': train_config['trainer_args']['lr_scheduler_type'],
                'output_dir': train_config['trainer_args']['output_dir'],
                'logging_dir': train_config['trainer_args']['logging_dir'],
                'weight_decay': train_config['trainer_args']['weight_decay'],
                'early_stopping_patience': train_config['trainer_args']['early_stopping_patience'],
            },

            'KoBERT': {
                'max_length': train_config['KoBERT']['max_length'],
            }
        }

    def save_model_config(self):
        save_config_path = self._get_config_path()
        os.makedirs(os.path.dirname(save_config_path), exist_ok=True)
        with open(save_config_path, 'w') as f:
            config_data = {attr: getattr(self, attr) for attr in dir(self) if
                           not callable(getattr(self, attr)) and not attr.startswith("_")}
            yaml.dump(config_data, f, default_flow_style=False)
