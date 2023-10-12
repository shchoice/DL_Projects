import logging
import os

from transformers import BertForSequenceClassification
from kobert_tokenizer import KoBERTTokenizer

from apps.src.config import constants
from apps.src.models.base_model_classifier import BaseModelClassifier
from apps.src.schemas.train_config import TrainConfig


class KoBERTClassifier:
    def __init__(self, num_labels, train_config: TrainConfig):
        self.model_card = constants.MODEL_KOBERT_CARD_NAME
        self.train_config = train_config
        self.cache_dir = os.path.join(self.train_config['base_dir'], constants.MODEL_PATH_NAME)
        self.logger = logging.getLogger(constants.LOGGER_INFO_NAME)
        self.tokenizer = None
        self.model = None
        self.load_model(num_labels)

    def load_model(self, num_labels):
        self.tokenizer = KoBERTTokenizer.from_pretrained(self.model_card, cache_dir=self.cache_dir)
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_card, num_labels=num_labels, cache_dir=self.cache_dir
        )

    def tokenize(self, examples):
        # def find_invalid_data(data_list):
        #     invalid_data = [(index, item) for index, item in enumerate(data_list) if not isinstance(item, str)]
        #     return invalid_data
        #
        # invalid_entries = find_invalid_data(examples['Text'])
        # if len(invalid_entries) > 0:
        #     self.logger.error(invalid_entries)
        #     self.logger.error(examples['Text'][330])
        tokenized_data = self.tokenizer(
            examples['Text'],
            truncation=True,
            max_length=self.train_config['KoBERT']['max_length'],
            padding=self.train_config['tokenizer']['padding']
        )

        return {
            "input_ids": tokenized_data["input_ids"],
            "token_type_ids": tokenized_data.get("token_type_ids", None),
            "attention_mask": tokenized_data["attention_mask"],
            "labels": examples["labels"]
        }