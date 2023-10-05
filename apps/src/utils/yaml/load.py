from typing import Any, Dict, Union

import yaml

from apps.src.schemas.preprocess_config import PreprocessConfig


def load_yaml_config(yaml_file: str) -> Dict[str, Any]:
    with open(yaml_file, 'r') as f:
        yaml_config = yaml.safe_load(f)

    return yaml_config


def load_data_config(yaml_file: str, schema: Union[PreprocessConfig]) -> Dict[str, Any]:
    with open(yaml_file, 'r') as f:
        data_config = yaml.safe_load(f)
        data_config['text_dataset'] = schema.text_dataset
        data_config['base_dir'] = schema.base_dir

    return data_config