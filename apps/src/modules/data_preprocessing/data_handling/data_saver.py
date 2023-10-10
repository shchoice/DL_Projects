import os
import shutil
from typing import Dict
import pandas as pd

from apps.src.config import constants


class DataSaver:
    @staticmethod
    def get_data_dir(data_config: Dict) -> str:
        return os.path.join(data_config['base_dir'], constants.DATA_PATH_NAME,
                            data_config['text_dataset'])

    @staticmethod
    def clear_and_create_directory(directory: str) -> None:
        shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory, exist_ok=True)

    @staticmethod
    def save_dataframe_to_csv(df: pd.DataFrame, filepath: str) -> None:
        df.to_csv(
            path_or_buf=filepath,
            index=False,
            sep=constants.DATA_COLUMN_SEP,
            header=None,
            encoding=constants.DATA_FILE_ENCODING
        )

    @classmethod
    def save_df_splitted(cls, df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame,
                         df_train_valid: pd.DataFrame, data_config: Dict) -> None:
        data_dir = cls.get_data_dir(data_config)

        path_names = [
            constants.DATA_TRAIN_PATH_NAME,
            constants.DATA_VALID_PATH_NAME,
            constants.DATA_TEST_PATH_NAME,
            constants.DATA_TRAIN_VALID_PATH_NAME
        ]

        filenames = [
            constants.DATA_TRAIN_PATH_NAME + data_config['filename_extension'],
            constants.DATA_VALID_PATH_NAME + data_config['filename_extension'],
            constants.DATA_TEST_PATH_NAME + data_config['filename_extension'],
            constants.DATA_TRAIN_VALID_PATH_NAME + data_config['filename_extension']
        ]

        dataframes = [df_train, df_valid, df_test, df_train_valid]

        for path_name, filename, df in zip(path_names, filenames, dataframes):
            full_path = os.path.join(data_dir, path_name)
            cls.clear_and_create_directory(full_path)
            file_path = os.path.join(full_path, filename)
            cls.save_dataframe_to_csv(df, file_path)
