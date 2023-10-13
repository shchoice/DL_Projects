from apps.src.modules.predict.predict_manager import PredictManager
from apps.src.schemas.predict_config import PredictConfig


class PredictService:
    def __init__(self, predict_config: PredictConfig):
        self.predict_config = predict_config
        self.predict_manager = PredictManager(predict_config)
        self.predictor = None

    def run_predict(self):
        predict_document_list = self.predict_config['documents']
        self.predictor = self.predict_manager.initialize_predictor()
        return self.predictor.predict(predict_document_list)

    def run_model_exchange(self):
        self.predictor = self.predict_manager.initialize_predictor()
        self.predictor.load_model_from_checkpoint()
