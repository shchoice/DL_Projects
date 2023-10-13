from apps.src.modules.predict.KoBERT_predictor import KoBERTPredictor


class PredictManager:
    def __init__(self, predict_config):
        self.predict_config = predict_config

    def initialize_predictor(self):
        if self.predict_config['model_type'] == 'KoBERT':
            return KoBERTPredictor(predict_config=self.predict_config, from_trainer=False)

        return None