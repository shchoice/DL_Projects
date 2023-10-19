import logging

from apps.src.config import constants
from apps.src.modules.training.trainer import Trainer
from apps.src.schemas.train_config import TrainConfig


class TrainingService:
    def __init__(self, train_config: TrainConfig):
        self.train_config = train_config
        self.trainer = Trainer(train_config)

        self.logger = logging.getLogger(constants.LOGGER_INFO_NAME)

    def run_classifier(self):
        dataset = self.trainer.prepare_dataset()
        trainer, test_dataset = self.trainer.train(dataset)
        self.trainer.evalutate(trainer, test_dataset)
        self.trainer.initialize_predictor(trainer.model)

        del trainer
