import os
from abc import ABC, abstractmethod

from apps.src.config import constants
from apps.src.schemas.train_schema import TrainSchema


class BaseModelClassifier(ABC):
    def __init__(self, num_labels: int, train_config: TrainSchema):
        self.tokenizer = None
        self.model = self.load_model(num_labels)
        self.train_config = train_config
        self.cache_dir = os.path.join(self.train_config['base_dir'], constants.MODEL_PATH_NAME)

    @abstractmethod
    def load_model(self, num_labels):
        pass

    @abstractmethod
    def tokenize(self, examples):
        pass
