import logging
import os
import pickle
from typing import Dict, Any, Type, Union

from datasets import DatasetDict
from sklearn.preprocessing import LabelEncoder

from apps.src.config import constants


class LabelEncoderManager:
    def __init__(self, config):
        self.label_encoder = LabelEncoder()
        self.config = config
        self.logger = logging.getLogger(constants.LOGGER_INFO_NAME)

    def encode_labels(self, dataset: DatasetDict) -> DatasetDict:
        for dataset_type in ['train', 'valid', 'test']:
            if dataset_type == 'train':
                self.label_encoder.fit(dataset[dataset_type]['Category'])
            encoded_labels = self.label_encoder.transform(dataset[dataset_type]['Category'])
            dataset[dataset_type] = dataset[dataset_type].map(
                self.add_encoded_labels,
                with_indices=True,
                fn_kwargs={'encoded_labels': encoded_labels}
            )

        self.save_label_encoder()

        return dataset

    @staticmethod
    def add_encoded_labels(example: dict, idx: int, encoded_labels: Union[list]):
        example['labels'] = int(encoded_labels[idx])

        return example

    def _get_label_encoder_path(self) -> str:
        return os.path.join(
            self.config['base_dir'],
            constants.OUTPUT_PATH_NAME,
            self.config['text_dataset'],
            self.config['model_type'],
            constants.LABEL_ENCODER_NAME
        )

    def save_label_encoder(self) -> None:
        save_path = self._get_label_encoder_path()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(file=save_path, mode='wb') as f:
            pickle.dump(self.label_encoder, f)
            self.logger.info('Saved label_encoder.pkl to %s', save_path)

    def load_label_encoder(self) -> Type[LabelEncoder]:
        load_path = self._get_label_encoder_path()

        if not os.path.exists(load_path):
            raise FileNotFoundError(f'LabelEncoder file not found at {load_path}')

        with open(file=load_path, mode='rb') as f:
            return pickle.load(f)
