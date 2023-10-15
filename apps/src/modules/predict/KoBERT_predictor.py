import os.path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from kobert_tokenizer import KoBERTTokenizer

from transformers import BertForSequenceClassification

from apps.src.config import constants
from apps.src.exception.model_exchange_exception import ModelExchangeException
from apps.src.modules.common.label_encoder_manager import LabelEncoderManager
from apps.src.modules.predict.base_predictor import BasePredictor


class KoBERTPredictor(BasePredictor):
    def __init__(self, trained_model=None, predict_config=None, from_trainer=False):
        super().__init__(predict_config)

        self.device = self._get_gpu_device()
        self.tokenizer = KoBERTTokenizer.from_pretrained(constants.MODEL_KOBERT_CARD_NAME)
        self.label_encoder = LabelEncoderManager(self.predict_config).load_label_encoder()

        # 1. TrainService가 학습이 끝나고 넘겨줄 경우
        if from_trainer is True:
            self._model = trained_model
            self._move_model_to_device()
        #2. 메모리에 model 이 없을 경우(재기동 등의 상황)
        elif self._model is None and predict_config:
            self.load_model_from_checkpoint()

    def _move_model_to_device(self):
        # 반드시 DataParallel은 적용하고, 그 후에 모델을 디바이스로 옮겨야함, 반대로 하면 다른 디바이스에 할당될 수 있음
        if self.multi_gpu:
            self._model = torch.nn.DataParallel(self._model)
        self._model.to(self.device)

    @BasePredictor.model.setter
    def model(self, checkpoint):
        if os.path.exists(checkpoint):
            print(torch.cuda.device_count())
            self._model = BertForSequenceClassification.from_pretrained(checkpoint)
            self._move_model_to_device()
            self.logger.info('New KoBERT model is loaded: %s', checkpoint)
        else:
            self.logger.error('Model path does not exist: %s', checkpoint)
            raise ModelExchangeException(f'Not found error: Check model checkpoint path, {checkpoint}')

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
            # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
            return torch.device(f"cuda:{gpu_ids[0]}")
        else:
            return torch.device(f"cuda:{gpu_id}")

    def predict(self, predict_document_list: List) -> Tuple[List[List[int]], List[List[float]]]:
        torch.cuda.empty_cache()
        inputs = self.tokenizer(predict_document_list, return_tensors="pt", padding=True, truncation=True,
                                max_length=self.predict_config['KoBERT']['max_length']).to(self.device)

        outputs = self._model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)

        top_k_values, top_k_indices = torch.topk(probs, self.predict_config['top_k'], dim=1)
        top_k_decoded_labels = [self.label_encoder.inverse_transform(indices.tolist()) for indices in top_k_indices]

        return top_k_decoded_labels, np.round(np.array(top_k_values.tolist()) * 100, 5).tolist()

    def load_model_from_checkpoint(self):
        checkpoint = self.get_model_path()
        # model setter 적용
        self.model = checkpoint

    def get_model_path(self):
        return os.path.join(self.predict_config['base_dir'], constants.OUTPUT_PATH_NAME,
                            self.predict_config['text_dataset'], 'KoBERT', self.predict_config['load_model_name'])
