import logging
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from kobert_tokenizer import KoBERTTokenizer


from apps.src.config import constants
from apps.src.modules.common.label_encoder_manager import LabelEncoderManager
from apps.src.modules.training.managers.model_manager import ModelManager


class KoBERTPredictor:
    def __init__(self, predict_config=None):
        self.predict_config = predict_config
        self.model_manager = ModelManager(predict_config)
        self.tokenizer = KoBERTTokenizer.from_pretrained(constants.MODEL_KOBERT_CARD_NAME)
        self.label_encoder = LabelEncoderManager(predict_config).load_label_encoder()
        self.multi_gpu = False
        self.device = self._get_gpu_device()

        self.logger = logging.getLogger(constants.LOGGER_INFO_NAME)

    def predict(self, predict_document_list: List) -> Tuple[List[List[int]], List[List[float]]]:
        model = self.model_manager.get_model_instance(mode='predict').model
        model.to(self.device)
        if self.multi_gpu:
            model = torch.nn.DataParallel(model)
        model.eval()

        inputs = self.tokenizer(predict_document_list, return_tensors="pt", padding=True, truncation=True,
                                max_length=self.predict_config['KoBERT']['max_length']).to(self.device)

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)

        raw_top_k_values, raw_top_k_indices = torch.topk(probs, self.predict_config['top_k'], dim=1)

        top_k_decoded_labels = [self.label_encoder.inverse_transform(indices.tolist()) for indices in raw_top_k_indices]
        top_k_values = np.round(np.array(raw_top_k_values.tolist()) * 100, 5).tolist()

        return top_k_decoded_labels, top_k_values

    def _get_gpu_device(self):
        gpu_id = self.predict_config['gpu_id']

        if gpu_id == 'cpu':
            return torch.device("cpu")

        elif self.predict_config['gpu_id'] == 'auto':
            if torch.cuda.device_count() > 1:
                self.multi_gpu = True
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif ',' in gpu_id:
            gpu_ids = [int(id_.strip()) for id_ in gpu_id.split(',')]
            if not all(isinstance(id_, int) and 0 <= id_ < torch.cuda.device_count() for id_ in gpu_ids):
                raise ValueError(f"Invalid gpu_id: {gpu_id}")

            self.multi_gpu = True if len(gpu_ids) > 1 else False
            return torch.device(f"cuda:{gpu_ids[0]}")
        else:
            return torch.device(f"cuda:{gpu_id}")