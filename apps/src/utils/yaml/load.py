import os
from typing import Any, Dict, Union, Type

import yaml

from apps.src.config import constants
from apps.src.exception.model_exchange_exception import ModelExchangeException
from apps.src.schemas.data_preprocess_schema import DataPreprocessSchema
from apps.src.schemas.reload_schema import ReloadSchema


def load_yaml_config(yaml_file: str) -> Dict[str, Any]:
    with open(yaml_file, 'r') as f:
        yaml_config = yaml.safe_load(f)

    return yaml_config


def load_data_config(yaml_file: str, schema: Union[DataPreprocessSchema]) -> Dict[str, Any]:
    with open(yaml_file, 'r') as f:
        data_config = yaml.safe_load(f)
        data_config['text_dataset'] = schema.text_dataset
        data_config['base_dir'] = schema.base_dir

    return data_config


def load_reload_model_config(schema: Type[ReloadSchema]) -> Dict[str, Any]:
    yaml_file = os.path.join(schema.base_dir, constants.MODEL_CONFIG_PATH_NAME,
                            schema.text_dataset, schema.model_type,
                            constants.MODEL_CONFIG_YAML_FILE_NAME)

    with open(yaml_file, 'r') as f:
        reload_config = yaml.safe_load(f)

        if reload_config['model_type'] != schema.model_type:
            raise ModelExchangeException(f"Check Input parameter {schema.model_type}, expect: {reload_config['model_type']}")
        elif reload_config['text_dataset'] != schema.text_dataset:
            raise ModelExchangeException(f"Check Input parameter {schema.text_dataset}, expect: {reload_config['text_dataset']}")
        elif reload_config['base_dir'] != schema.base_dir:
            raise ModelExchangeException(f"Check Input parameter {schema.base_dir}, expect: {reload_config['base_dir']}")

        reload_config['load_model_name'] = schema.load_model_name

    return reload_config
