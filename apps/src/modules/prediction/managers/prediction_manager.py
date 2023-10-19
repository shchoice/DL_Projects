from apps.src.modules.prediction.predictors.KoBERT_predictor import KoBERTPredictor


class PredictionManager:
    _predictors = {}

    def __init__(self, predict_config):
        self.predict_config = predict_config
        self.predictor = None

    def initialize_predictor(self):
        model_type = self.predict_config['model_type']

        if model_type in PredictionManager._predictors:
            return PredictionManager._predictors[model_type]

        predictor = self._get_predictor(model_type)
        if predictor:
            PredictionManager._predictors[model_type] = predictor

        return predictor

    def _get_predictor(self, model_type):
        if model_type == 'KoBERT':
            return KoBERTPredictor(predict_config=self.predict_config, from_trainer=False)

    def load_trained_model(self, predictor, new_model_name):
        self.predict_config['load_model_name'] = new_model_name
        predictor.load_model_from_checkpoint()
