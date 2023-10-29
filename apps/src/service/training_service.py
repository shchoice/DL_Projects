import logging

from apps.src.config import constants
from apps.src.modules.training.model_config.base_config import BaseConfig
from apps.src.modules.training.trainer import Trainer


class TrainingService:
    def __init__(self, train_config: BaseConfig):
        self.train_config = train_config
        self.trainer = Trainer(train_config)

        self.logger = logging.getLogger(constants.LOGGER_INFO_NAME)

    def run_classifier(self):
        dataset = self.trainer.prepare_dataset()
        trainer, test_dataset = self.trainer.train(dataset)
        self.trainer.evalutate(trainer, test_dataset)

        del trainer
