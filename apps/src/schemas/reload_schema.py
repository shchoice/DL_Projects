from pydantic import BaseModel


class ReloadSchema(BaseModel):
    model_type: str
    text_dataset: str
    base_dir: str
    load_model_name: str
