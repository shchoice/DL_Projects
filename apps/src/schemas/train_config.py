from pydantic import BaseModel


class TrainConfig(BaseModel):
    model_type: str
