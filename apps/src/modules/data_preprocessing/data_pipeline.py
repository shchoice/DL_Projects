import os

from apps.src.config import constants
from apps.src.modules.data_preprocessing.cleaning.data_cleaning import DataCleaning
from apps.src.modules.data_preprocessing.data_handling.data_handler import DataHandler
from apps.src.modules.data_preprocessing.tokenizing.data_tokenizer import DataTokenizer
from apps.src.schemas.data_preprocess_config import DataPreprocessConfig


class DataPipeline:
    def __init__(self, data_config: DataPreprocessConfig):
        self.data_config = data_config
        self.dataframe = None

    def get_tsv_files_path(self) -> str:
        return os.path.join(self.data_config['base_dir'],
                            constants.DATA_PATH_NAME,
                            self.data_config['text_dataset'],
                            constants.DATA_RAW_PATH_NAME)

    def read_text(self):
        data_handler = DataHandler()
        tsv_files_path = self.get_tsv_files_path()

        tsv_files_list = data_handler.read_tsv_files(tsv_files_path, self.data_config['filename_extension'])
        self.dataframe = data_handler.convert_tsv_to_df(tsv_files_list, self.data_config['column_name'])

    def cleaning(self):
        data_cleaning = DataCleaning(self.data_config)
        self.dataframe = data_cleaning.clean_df(self.dataframe, self.data_config)

    def augmentation(self):
        data_handler = DataHandler()

        self.dataframe = data_handler.augment_data(self.dataframe)

    def tokenizing(self):
        data_tokeinzer = DataTokenizer()
        self.dataframe = data_tokeinzer.tokenize_df(self.dataframe)

    def save_preprocessed_data(self):
        data_handler = DataHandler()
        data_handler.save_df_to_splitted_tsv(self.dataframe, self.data_config)
