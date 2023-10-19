import logging

from transformers import logging as transformers_logging, DataCollatorWithPadding

from apps.src.config import constants
from apps.src.modules.common.label_encoder_manager import LabelEncoderManager
from apps.src.modules.prediction.predictors.KoBERT_predictor import KoBERTPredictor
from apps.src.modules.training.managers.data_load_manager import DataLoadManager
from apps.src.modules.training.util.early_stopping import SaveLastModelCallback
from apps.src.modules.training.managers.metrics_manager import MetricsManager
from apps.src.modules.training.managers.model_manager import ModelManager
from apps.src.modules.training.trainer_with_logger import TrainerWithLogger
from apps.src.modules.training.managers.training_config_manager import TrainingConfigManager
from apps.src.schemas.train_config import TrainConfig


class Trainer:
    def __init__(self, train_config: TrainConfig):
        self.train_config = train_config
        self.model_manager = ModelManager(train_config)
        self.training_config_manager = TrainingConfigManager(train_config)
        self.metrics_manager = MetricsManager()
        self.data_load_manager = DataLoadManager(self.train_config)
        self.label_encoder_manager = LabelEncoderManager(self.train_config)

        self.logger = logging.getLogger(constants.LOGGER_INFO_NAME)

    def prepare_dataset(self):
        dataset = self.data_load_manager.load_dataset()
        dataset = self.label_encoder_manager.encode_labels(dataset)
        return dataset

    def train(self, dataset):
        num_labels = dataset['train'].to_pandas()['Category'].nunique()
        classifier_model = self.model_manager.initialize_model(num_labels)
        if classifier_model is None:
            raise ValueError("Invalid model type specified")

        self.logger.info('Text tokenzing starts!')
        dataset = dataset.map(classifier_model.tokenize, batched=True)
        self.logger.info('Text tokenzing finished!')

        data_collator = DataCollatorWithPadding(tokenizer=classifier_model.tokenizer)
        early_stopping_callback = \
            SaveLastModelCallback(early_stopping_patience=self.train_config['trainer_args']['early_stopping_patience'])

        trainer = TrainerWithLogger(
            model=classifier_model.model,
            args=self.training_config_manager.get_training_arguments(),
            train_dataset=dataset["train"],
            eval_dataset=dataset["valid"],
            tokenizer=classifier_model.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.metrics_manager.compute_metrics,
            callbacks=[early_stopping_callback],
            tqdm_logger=self.logger,
        )
        early_stopping_callback.set_trainer(trainer)
        trainer.train()
        self.logger.info("Training finished!")

        self.model_manager.update_and_save_config()
        self.logger.info("Model Config File Updated and saved!!!")

        test_prediction = trainer.evaluate(dataset['test'])
        self.logger.info('Test dataset validation result: %s', test_prediction)

        return trainer, dataset['test']

    def evalutate(self, trainer, test_dataset):
        test_prediction = trainer.evaluate(test_dataset)
        self.logger.info('Test dataset validation result: %s', test_prediction)

    def initialize_predictor(self, model):
        if self.train_config['model_type'] == 'KoBERT':
            KoBERTPredictor(trained_model=model, predict_config=self.train_config, from_trainer=True)

    @classmethod
    def configure_logging_for_transformers(cls, main_logger):
        transformers_logger = transformers_logging.get_logger("transformers")

        for handler in main_logger.handlers:
            transformers_logger.addHandler(handler)

        transformers_logger.setLevel(logging.INFO)
        transformers_logger.propagate = False
