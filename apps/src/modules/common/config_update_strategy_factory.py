from apps.src.modules.prediction.prediction_config_update_strategy import PredictionConfigUpdateStrategy
from apps.src.modules.training.training_config_update_strategy import TrainingConfigUpdateStrategy


class ConfigUpdateStrategyFactory:
    @staticmethod
    def get_strategy(model_type: str):
        if model_type == 'Training':
            return TrainingConfigUpdateStrategy()
        elif model_type == 'Prediction':
            return PredictionConfigUpdateStrategy()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
