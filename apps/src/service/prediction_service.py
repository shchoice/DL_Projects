from apps.src.modules.prediction.managers.prediction_manager import PredictionManager
from apps.src.modules.training.model_config.base_config import BaseConfig


class PredictionService:
    def __init__(self, predict_config: BaseConfig):
        self.predict_config = predict_config
        self.prediction_manager = PredictionManager(predict_config)

    def run_predict(self):
        predict_document_list = self.predict_config['documents']
        predictor = self.prediction_manager.get_model_instance()

        top_k_decoded_labels, top_k_values = predictor.predict(predict_document_list)

        return top_k_decoded_labels, top_k_values

    def run_model_exchange(self):
        self.prediction_manager.load_trained_model()
