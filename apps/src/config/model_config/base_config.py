import copy
import os
from typing import Any, List

import yaml

from apps.src.config import constants


class BaseConfig:
    _instance = None
    _default_config = {}

    def __new__(cls, model_type: str, base_dir: str, text_dataset: str):
        if cls._instance is None:
            cls._instance = super(BaseConfig, cls).__new__(cls)
            cls._instance._initialize_config(model_type, base_dir, text_dataset)
        return cls._instance

    def _initialize_config(self, model_type: str, base_dir: str, text_dataset: str):
        self.model_config = self._default_config.copy()
        self.model_config['model_type'] = model_type
        self.model_config['base_dir'] = base_dir
        self.model_config['text_dataset'] = text_dataset

        config_path = self._get_config_path()

        # 첫번째 model_type, text_dataset 사용 시에는 config.yaml 파일에서 읽어오지 않음(존재X), 2번째 부터 참조
        if os.path.isfile(config_path):
            self._load_config_from_file(config_path)

    def _get_config_path(self) -> str:
        return os.path.join(self.model_config['base_dir'], constants.MODEL_CONFIG_PATH_NAME,
                            self.model_config['text_dataset'], self.model_config['model_type'], 'config.yaml')

    def _load_config_from_file(self, config_path: str):
        with open(config_path, 'r') as f:
            self.model_config.update(yaml.safe_load(f))

    @classmethod
    def get_model_config(cls):
        return copy.deepcopy(cls._instance.model_config)

    def set_model_config(self, key_path: List[str], value: Any):
        config = self.model_config
        for key in key_path[:-1]:
            config = config.get(key, {})
        config[key_path[-1]] = value

    def save_model_config(self, config):
        save_config_path = self._get_config_path()
        os.makedirs(os.path.dirname(save_config_path), exist_ok=True)
        with open(save_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
