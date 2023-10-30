from apps.src.modules.prediction.predictor_factory import PredictorFactory
from apps.src.utils.pattern.singleton import SingletonMeta


class PredictionManager(metaclass=SingletonMeta):
    _instances = {}

    def __init__(self, config):
        self.config = config

    def get_model_instance(self):
        key = (self.config['model_type'], self.config['base_dir'], self.config['text_dataset'])
        if key not in PredictionManager._instances or self.config['load_trained_model']:
            PredictionManager._instances[key] = PredictorFactory.create_predictor(self.config)

        return PredictionManager._instances[key]
