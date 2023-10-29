from pydantic import BaseModel

class DataPreprocessSchema(BaseModel):
    text_dataset: str
    base_dir: str
