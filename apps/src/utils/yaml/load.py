import os.path
from typing import Any, Type, Dict

import yaml


def load_yaml_config(yaml_file: str) -> Dict[str, Any]:
    with open(yaml_file, 'r') as f:
        yaml_config = yaml.safe_load(f)

    return yaml_config

def load_data_config(yaml_file: str, schema: Type) -> Dict[str, Any]:
    with open(yaml_file, 'r') as f:
        data_config = yaml.safe_load(f)
        data_config['data']['collection'] = schema.collection
        data_config['data']['base_dir'] = schema.base_dir

    return data_config