from pydantic import BaseModel

class DataPreprocessConfig(BaseModel):
    text_dataset: str
    base_dir: str
