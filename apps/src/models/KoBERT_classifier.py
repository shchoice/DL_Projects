import logging
import os

from transformers import BertForSequenceClassification
from kobert_tokenizer import KoBERTTokenizer

from apps.src.config import constants
from apps.src.modules.training.model_config.base_config import BaseConfig


class KoBERTClassifier:
    def __init__(self, num_labels, config: BaseConfig, mode: str):
        self.model_card = constants.MODEL_KOBERT_CARD_NAME
        self.config = config
        self.cache_dir = os.path.join(self.config['base_dir'], constants.MODEL_PATH_NAME)
        self.logger = logging.getLogger(constants.LOGGER_INFO_NAME)
        self.tokenizer = None
        self.model = None
        self.load_model(num_labels, mode)

    def load_model(self, num_labels, mode='train'):
        self.tokenizer = KoBERTTokenizer.from_pretrained(self.model_card, cache_dir=self.cache_dir)

        checkpoint = self.get_model_path() if self.config['load_trained_model'] else self.model_card

        if mode == 'train':
            self.model = BertForSequenceClassification.from_pretrained(
                checkpoint, num_labels=num_labels, cache_dir=self.cache_dir
            )
        elif mode == 'predict':
            self.model = BertForSequenceClassification.from_pretrained(checkpoint)

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def tokenize(self, examples):
        tokenized_data = self.tokenizer(
            examples['Text'],
            truncation=True,
            max_length=self.config['KoBERT']['max_length'],
            padding=self.config['tokenizer']['padding']
        )

        return {
            "input_ids": tokenized_data["input_ids"],
            "token_type_ids": tokenized_data.get("token_type_ids", None),
            "attention_mask": tokenized_data["attention_mask"],
            "labels": examples["labels"]
        }

    def get_model_path(self):
        return os.path.join(self.config['base_dir'], constants.OUTPUT_PATH_NAME,
                            self.config['text_dataset'], 'KoBERT', self.config['load_model_name'])
