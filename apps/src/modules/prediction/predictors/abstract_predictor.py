import logging
from abc import ABC, abstractmethod
from typing import List, Tuple

from apps.src.config import constants


class AbstractPredictor(ABC):
    '''
    추상화 수준에 대한 가설정입니다. 클래스들을 추가하면서 추가/변경될 수 있기 때문에,
    현재는 추상 클래스만 생성해두고 상속을 사용하지 않았습니다.
    '''

    def __init__(self, predict_config):
        self.tokenizer = None
        self.multi_gpu = False
        self.predict_config = predict_config
        self.logger = logging.getLogger(constants.LOGGER_INFO_NAME)

    @abstractmethod
    # @property
    def model(self):
        return AbstractPredictor._model

    @abstractmethod
    # @model.setter
    def model(self, path):
        raise NotImplementedError("You need to implement this in child classes!")

    @abstractmethod
    def predict(self, predict_document_list: List) -> Tuple[List[List[int]], List[List[float]]]:
        raise NotImplementedError("You need to implement this in child classes!")

    @abstractmethod
    def load_model_from_checkpoint(self):
        pass

    @abstractmethod
    def _get_gpu_device(self):
        pass
