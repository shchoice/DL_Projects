import logging
import os

from apps.src.config import constants
from apps.src.schemas.train_schema import TrainSchema
from datasets import load_dataset, DatasetDict


class DataLoadManager:
    def __init__(self, train_config: TrainSchema):
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
