from apps.src.modules.common.config_factory import ConfigFactory
from apps.src.modules.training.model_config.base_config import BaseConfig
from apps.src.utils.pattern.singleton import SingletonMeta


class ConfigManager(metaclass=SingletonMeta):
    _instances = {}

    @classmethod
    def get_config_instance(cls, model_type: str, base_dir: str, text_dataset: str) -> BaseConfig:
        key = (model_type, base_dir, text_dataset)
        if key not in cls._instances:
            cls._instances[key] = ConfigFactory.create_config(model_type, base_dir, text_dataset)

        return cls._instances[key]
