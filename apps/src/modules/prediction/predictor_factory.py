from apps.src.modules.prediction.predictors.KoBERT_predictor import KoBERTPredictor
from apps.src.modules.training.model_config.base_config import BaseConfig


class PredictorFactory:
    @staticmethod
    def create_predictor(config: BaseConfig):
        if config['model_type'] == 'KoBERT':
            return KoBERTPredictor(predict_config=config)
        else:
            raise ValueError('Unknown model type: %s'.format(config['model_type']))
