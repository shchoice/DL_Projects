import logging

from transformers import DataCollatorWithPadding
from transformers import logging as transformers_logging

from apps.src.config import constants
from apps.src.modules.common.label_encoder_manager import LabelEncoderManager
from apps.src.modules.predict.KoBERT_predictor import KoBERTPredictor
from apps.src.modules.train.classifier_trainer import ClassifierTrainer
from apps.src.modules.train.data_load_manager import DataLoadManager
from apps.src.modules.train.early_stopping import SaveLastModelCallback
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
        self.data_loader = DataLoadManager(self.train_config)
        self.label_encoder_manager = LabelEncoderManager(self.train_config)

        self.logger = logging.getLogger(constants.LOGGER_INFO_NAME)
        self.configure_logging_for_transformers(self.logger)

    def run_classifier(self):
        # 1. 데이터 로딩
        dataset = self.data_loader.load_dataset()
        dataset = self.label_encoder_manager.encode_labels(dataset)

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
        early_stopping_callback = \
            SaveLastModelCallback(early_stopping_patience=self.train_config['trainer_args']['early_stopping_patience'])

        trainer = ClassifierTrainer(
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

        # 4. 모델 학습 정보 저장
        self.model_manager.update_and_save_config()
        self.logger.info("Model Config File Updated and saved!!!")

        # 5. 검증 및 평가
        test_prediction = trainer.evaluate(dataset['test'])
        self.logger.info('Test dataset validation result: %s', test_prediction)

        KoBERTPredictor(trained_model=trainer.model, predict_config=self.train_config, from_trainer=True)

        del trainer

    @classmethod
    def configure_logging_for_transformers(cls, main_logger):
        transformers_logger = transformers_logging.get_logger("transformers")

        for handler in main_logger.handlers:
            transformers_logger.addHandler(handler)

        transformers_logger.setLevel(logging.INFO)
        transformers_logger.propagate = False
