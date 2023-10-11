from typing import Dict, Any

from transformers import Trainer

class ClassifierTrainer(Trainer):
    def __init__(self, tqdm_logger, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tqdm_logger = tqdm_logger

    def log(self, logs: Dict[str, Any]) -> None:
        super().log(logs)
        for key, value in logs.items():
            self.tqdm_logger.info(f"{key}: {value}")

