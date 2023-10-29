from apps.src.modules.training.model_config.KoBERT_config import KoBERTConfig


class ConfigFactory:
    @staticmethod
    def create_config(model_type: str, base_dir: str, text_dataset: str):
        if model_type == 'KoBERT':
            return KoBERTConfig(model_type, base_dir, text_dataset)
        else:
            raise ValueError(f'Unknown model type: {model_type}')
