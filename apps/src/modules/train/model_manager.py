from apps.src.models.KoBERT_classifier import KoBERTClassifier


class ModelManager:
    def __init__(self, train_config):
        self.train_config = train_config

    def initialize_model(self, num_labels):
        if self.train_config['model_type'] == 'KoBERT':
            num_labels = num_labels
            return KoBERTClassifier(num_labels=num_labels, train_config=self.train_config)

        return None

