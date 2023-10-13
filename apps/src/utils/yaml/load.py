import os
from typing import Any, Dict, Union

import yaml

from apps.src.config import constants
from apps.src.schemas.data_preprocess_config import DataPreprocessConfig
from apps.src.schemas.predict_config import PredictConfig
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

def load_predict_config(schema: Union[PredictConfig]) -> Dict[str, Any]:
    yaml_file = os.path.join(schema.base_dir, constants.MODEL_CONFIG_PATH_NAME, schema.text_dataset, schema.model_type, 'config.yaml')
    with open(yaml_file, 'r') as f:
        predict_config = yaml.safe_load(f)
        predict_config['model_type'] = schema.model_type
        predict_config['text_dataset'] = schema.text_dataset
        predict_config['base_dir'] = schema.base_dir
        predict_config['top_k'] = schema.top_k
        predict_config['documents'] = schema.documents
        predict_config['gpu_id'] = schema.gpu_id
        predict_config['load_trained_model'] = schema.load_trained_model
        predict_config['load_model_name'] = schema.load_model_name

    return predict_config

