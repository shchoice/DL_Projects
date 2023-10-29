from typing import Optional, List

from pydantic import BaseModel

from apps.src.config import constants


class PredictSchema(BaseModel):
    model_type: str
    text_dataset: str
    base_dir: str
    top_k: int
    documents: List[str] = []
    gpu_id: str
    load_trained_model: Optional[bool] = True
    load_model_name: Optional[str] = constants.MODEL_KOBERT_FINAL
