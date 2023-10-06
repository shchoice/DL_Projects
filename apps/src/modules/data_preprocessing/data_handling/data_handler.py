import logging
import os
import shutil
from typing import List, Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from apps.src.config import constants
from apps.src.modules.data_preprocessing.data_handling.data_saver import DataSaver


class DataHandler:
    def __init__(self):
        self.logger = logging.getLogger(constants.LOGGER_INFO_NAME)

    @staticmethod
    def read_tsv_files(tsv_files_path: str, filename_extension: str) -> List[str]:
        if not os.path.exists(tsv_files_path):
            raise Exception(f'TSV files does not exist: {tsv_files_path}')

        return [os.path.join(tsv_files_path, f) for f in os.listdir(tsv_files_path) if f.endswith(filename_extension)]

    @staticmethod
    def convert_tsv_to_df(tsv_files: str, df_columns: List[str]) -> pd.DataFrame:
        df_all = pd.DataFrame()

        for tsv_file in tsv_files:
            df_temp = pd.read_csv(tsv_file, sep=constants.DATA_COLUMN_SEP, header=None, encoding=constants.DATA_FILE_ENCODING)
            df_all = pd.concat([df_all, df_temp])

        df_all.columns = df_columns

        return df_all

    @staticmethod
    def augment_data(df_train: pd.DataFrame) -> pd.DataFrame:
        return df_train

    def df_split(self, df_all: pd.DataFrame, data_config: Dict, train_frac: int = 0.8) \
            -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        assert train_frac >= 0 and train_frac <= 1

        df_train, df_temp = train_test_split(
            df_all,
            train_size=train_frac,
            stratify=df_all['Category'],
            random_state=data_config['random_state'],
        )

        df_valid, df_test = train_test_split(
            df_temp,
            train_size=data_config['valid_test_ratio'],
            stratify=df_temp['Category'],
            random_state=data_config['random_state'],
        )

        for data_type, df in zip(['전체', 'Train', 'Valid', 'Test'], [df_all, df_train, df_valid, df_test]):
            self.logger.info(f"{data_type} category 개수 :", df['Category'].nunique(), '개')
            self.logger.info(f"{data_type} 데이터 개수 :", len(df), 'doc(s)')

        return df_train, df_valid, df_test

    def save_df_to_splitted_tsv(self, df: pd.DataFrame, data_config: Dict) -> None:
        df_train, df_valid, df_test = self.df_split(df, data_config, train_frac=data_config['train_ratio'])

        df_train_valid = pd.concat([df_train, df_valid])

        DataSaver.save_df_splitted(df_train, df_valid, df_train_valid, df_test, data_config)
        self.logger.info("Finsished creating a train/balid/test Dataset TSV file ")
