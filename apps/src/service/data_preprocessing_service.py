from apps.src.modules.data_preprocessing.data_pipeline import DataPipeline
from apps.src.schemas.data_preprocess_schema import DataPreprocessSchema


class DataPreprocessService:
    def __init__(self, data_config: DataPreprocessSchema):
        self.data_pipeline: DataPipeline = DataPipeline(data_config)

    def run_preprocess(self):
        self.data_pipeline.read_text()
        self.data_pipeline.cleaning()
        self.data_pipeline.augmentation()
        self.data_pipeline.tokenizing()
        self.data_pipeline.save_preprocessed_data()
