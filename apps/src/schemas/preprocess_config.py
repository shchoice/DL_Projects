from pydantic import BaseModel

class PreprocessConfig(BaseModel):
    text_dataset: str
    base_dir: str
