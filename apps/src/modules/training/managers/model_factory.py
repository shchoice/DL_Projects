from apps.src.models.KoBERT_classifier import KoBERTClassifier
from apps.src.modules.training.model_config.base_config import BaseConfig


class ModelFactory:
    @staticmethod
    def create_model(num_labels: int, config: BaseConfig, mode='train'):
        if config['model_type'] == 'KoBERT':
            return KoBERTClassifier(num_labels=num_labels, config=config, mode=mode)
        else:
            raise ValueError('Unknown model type: %s'.format(config['model_type']))
