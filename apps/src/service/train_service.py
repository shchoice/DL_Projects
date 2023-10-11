import logging
import os
import shutil

from transformers import DataCollatorWithPadding, EarlyStoppingCallback
from transformers import logging as transformers_logging


from apps.src.config import constants
from apps.src.modules.train.classifier_trainer import ClassifierTrainer
from apps.src.modules.train.data_loader import DataLoader
from apps.src.modules.train.metrics_manager import MetricsManager
from apps.src.modules.train.model_manager import ModelManager
from apps.src.modules.train.training_config_manager import TrainingConfigManager
from apps.src.schemas.train_config import TrainConfig


class TrainService:
    def __init__(self, train_config: TrainConfig):
        self.train_config = train_config
        self.model_manager = ModelManager(train_config)
        self.training_config_manager = TrainingConfigManager(train_config)
        self.metrics_manager = MetricsManager()
        self.logger = logging.getLogger(constants.LOGGER_INFO_NAME)
        self.configure_logging_for_transformers(self.logger)

    def run_classifier(self):
        # 1. 데이터 로딩
        data_loader = DataLoader(self.train_config)
        dataset = data_loader.load_dataset()
        dataset = data_loader.encode_labels(dataset)

        # 2. 모델 로딩
        num_labels = dataset['train'].to_pandas()['Category'].nunique()
        classifier_model = self.model_manager.initialize_model(num_labels)
        if classifier_model is None:
            raise ValueError("Invalid model type specified")

        # 3. Tokenize
        self.logger.info('Text tokenzing starts!')
        dataset = dataset.map(classifier_model.tokenize, batched=True)
        self.logger.info('Text tokenzing finished!')


        # 3. 모델 학습
        data_collator = DataCollatorWithPadding(tokenizer=classifier_model.tokenizer)

        trainer = ClassifierTrainer(
            model=classifier_model.model,
            args=self.training_config_manager.get_training_arguments(),
            train_dataset=dataset["train"],
            eval_dataset=dataset["valid"],
            tokenizer=classifier_model.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.metrics_manager.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.train_config['trainer_args']['early_stopping_patience'])],
            tqdm_logger=self.logger,
        )
        trainer.train()

        best_checkpoint = trainer.state.best_model_checkpoint
        final_path = os.path.join(os.path.dirname(best_checkpoint), "KoBERT_final")
        if os.path.exists(final_path):
            shutil.rmtree(final_path)
        shutil.copytree(best_checkpoint, final_path)

        self.logger.info("Training finished!")

        # 4. 검증 및 평가
        test_prediction = trainer.evaluate(dataset['test'])
        self.logger.info('Test dataset validation result: %s', test_prediction)

    @classmethod
    def configure_logging_for_transformers(cls, main_logger):
        transformers_logger = transformers_logging.get_logger("transformers")

        for handler in main_logger.handlers:
            transformers_logger.addHandler(handler)

        transformers_logger.setLevel(logging.INFO)
        transformers_logger.propagate = False
