import logging
import os
from typing import Union

from sklearn.preprocessing import LabelEncoder

from apps.src.config import constants
from apps.src.schemas.train_config import TrainConfig
from datasets import load_dataset, DatasetDict


class DataLoader:
    def __init__(self, train_config: TrainConfig):
        self.train_config = train_config
        self.logger = logging.getLogger(constants.LOGGER_INFO_NAME)

    def get_dataset_file_path(self, dataset_type: str) -> str:
        dataset_path = os.path.join(self.train_config['base_dir'], constants.DATA_PATH_NAME, self.train_config['text_dataset'])
        return os.path.join(dataset_path, dataset_type, dataset_type) + self.train_config['filename_extension']

    def load_dataset(self) -> DatasetDict:
        dataset_files = {
            dataset_type: self.get_dataset_file_path(dataset_type) for dataset_type in ['train', 'valid', 'test']
        }

        self.logger.info('Read datasets')
        dataset = load_dataset(
            'csv',
            data_files=dataset_files,
            delimiter=self.train_config['dataloader']['delimiter'],
            column_names=self.train_config['dataloader']['column_names']
        )
        self.logger.info('Dataset load results: %s', dataset)

        return dataset

    def encode_labels(self, dataset: DatasetDict):
        label_encoder = LabelEncoder()

        for dataset_type in ['train', 'valid', 'test']:
            encoded_labels = label_encoder.fit_transform(dataset[dataset_type]['Category']) if dataset_type == 'train' \
                else label_encoder.transform(dataset[dataset_type]['Category'])
            dataset[dataset_type] = dataset[dataset_type].map(
                self.add_encoded_labels,
                with_indices=True,
                fn_kwargs={'encoded_labels': encoded_labels}
            )

        return dataset

    @staticmethod
    def add_encoded_labels(example: dict, idx: int, encoded_labels: Union[list]):
        example['labels'] = int(encoded_labels[idx])
        return example
