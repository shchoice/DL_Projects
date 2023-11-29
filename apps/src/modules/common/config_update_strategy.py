from abc import ABC, abstractmethod


class ConfigUpdateStrategy(ABC):
    @abstractmethod
    def update_configuration(self, config, schema):
        pass
