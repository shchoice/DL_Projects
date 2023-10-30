from apps.src.modules.training.managers.model_factory import ModelFactory
from apps.src.utils.pattern.singleton import SingletonMeta


class ModelManager(metaclass=SingletonMeta):
    _classifier_instances = {}

    def __init__(self, config):
        self.config = config

    def _create_key(self):
        return (self.config['model_type'], self.config['base_dir'], self.config['text_dataset'])

    def get_model_instance(self, mode='train', num_labels: int = None):
        key = self._create_key()
        if key not in ModelManager._classifier_instances or self.config['load_trained_model']:
            ModelManager._classifier_instances[key] = ModelFactory.create_model(num_labels, self.config, mode)

        return ModelManager._classifier_instances[key]

    def update_model_instance(self, model, mode='train'):
        key = self._create_key()
        if mode == 'train':
            if key in ModelManager._classifier_instances:
                ModelManager._classifier_instances[key].model = model
                ModelManager._classifier_instances[key].model.load_model(num_labels=None, mode='predict')
            else:
                raise KeyError(f"Key {self.config['model_type']}, {self.config['base_dir']}, {self.config['text_dataset']} "
                               f"not found in the model instances.")
        elif mode == 'predict':
            ModelManager._classifier_instances[key].model = model
            ModelManager._classifier_instances[key].model.load_model(num_labels=None, mode='predict')
