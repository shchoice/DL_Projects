from typing import Any, Dict, Union

import yaml

from apps.src.schemas.data_preprocess_config import DataPreprocessConfig
from apps.src.schemas.train_config import TrainConfig


def load_yaml_config(yaml_file: str) -> Dict[str, Any]:
    with open(yaml_file, 'r') as f:
        yaml_config = yaml.safe_load(f)

    return yaml_config


def load_data_config(yaml_file: str, schema: Union[DataPreprocessConfig]) -> Dict[str, Any]:
    with open(yaml_file, 'r') as f:
        data_config = yaml.safe_load(f)
        data_config['text_dataset'] = schema.text_dataset
        data_config['base_dir'] = schema.base_dir

    return data_config


def load_train_config(yaml_file: str, schema: Union[TrainConfig]) -> Dict[str, Any]:
    with open(yaml_file, 'r') as f:
        train_config = yaml.safe_load(f)
        train_config['model_type'] = schema.model_type
        train_config['text_dataset'] = schema.text_dataset
        train_config['base_dir'] = schema.base_dir
        train_config['gpu_id'] = schema.gpu_id
        train_config['load_trained_model'] = schema.load_trained_model
        train_config['load_model_name'] = schema.load_model_name

    return train_config
