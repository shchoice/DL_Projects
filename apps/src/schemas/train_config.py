from typing import Optional

from pydantic import BaseModel

from apps.src.config import constants


class TrainConfig(BaseModel):
    model_type: str
    text_dataset: str
    base_dir: str
    gpu_id: str
    load_trained_model: Optional[bool] = False
    load_model_name: Optional[str] = constants.MODEL_KOBERT_FINAL
