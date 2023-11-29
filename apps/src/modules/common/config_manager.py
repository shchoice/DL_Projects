import threading

from apps.src.modules.common.config_factory import ConfigFactory
from apps.src.modules.common.config_update_strategy import ConfigUpdateStrategy
from apps.src.modules.common.config_update_strategy_factory import ConfigUpdateStrategyFactory
from apps.src.modules.training.model_config.base_config import BaseConfig
from apps.src.utils.pattern.singleton import SingletonMeta


class ConfigManager(metaclass=SingletonMeta):
    _instances = {}
    update_strategy = None
    _lock = threading.Lock()

    @classmethod
    def get_config_instance(cls, model_type: str, base_dir: str, text_dataset: str) -> BaseConfig:
        key = (model_type, base_dir, text_dataset)
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = ConfigFactory.create_config(model_type, base_dir, text_dataset)

            return cls._instances[key]

    @classmethod
    def set_update_strategy(cls, strategy: ConfigUpdateStrategy):
        with cls._lock:
            cls.update_strategy = strategy

    @classmethod
    def update_configuration(cls, config, schema):
        with cls._lock:
            if cls.update_strategy:
                cls.update_strategy.update_configuration(config, schema)

    @classmethod
    def configure(cls, config_type: str, schema):
        strategy = ConfigUpdateStrategyFactory.get_strategy(config_type)
        cls.set_update_strategy(strategy)
        config = cls.get_config_instance(schema.model_type, schema.base_dir, schema.text_dataset)
        cls.update_configuration(config, schema)

        return config
