import logging
from typing import List, Tuple

from apps.src.config import constants


class BasePredictor:
    _instances = {}
    _model = None

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(BasePredictor, cls).__new__(cls)
        return cls._instances[cls]

    def __init__(self, predict_config):
        self.tokenizer = None
        self.multi_gpu = False
        self.predict_config = predict_config
        self.logger = logging.getLogger(constants.LOGGER_INFO_NAME)

    @property
    def model(self):
        return BasePredictor._model

    @model.setter
    def model(self, path):
        raise NotImplementedError("You need to implement this in child classes!")

    def predict(self, predict_document_list: List) -> Tuple[List[List[int]], List[List[float]]]:
        raise NotImplementedError("You need to implement this in child classes!")

    def load_model_from_checkpoint(self):
        pass

    def _get_gpu_device(self):
        pass
