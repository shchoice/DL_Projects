from apps.src.modules.prediction.managers.prediction_manager import PredictionManager
from apps.src.modules.training.managers.model_manager import ModelManager
from apps.src.schemas.predict_config import PredictConfig


class PredictionService:
    def __init__(self, predict_config: PredictConfig):
        self.predict_config = predict_config
        self.prediction_manager = PredictionManager(predict_config)
        self.predictor = None

        self.model_manager = ModelManager(predict_config)

    def run_predict(self):
        predict_document_list = self.predict_config['documents']
        self.predictor = self.prediction_manager.initialize_predictor()
        if self.predict_config['load_trained_model'] is True:
            self.prediction_manager.load_trained_model(self.predictor, self.predict_config['load_model_name'])
        return self.predictor.predict(predict_document_list)

    def run_model_exchange(self):
        self.predictor = self.prediction_manager.initialize_predictor()
        self.predictor.load_model_from_checkpoint()

        self.model_manager.update_and_save_config()
