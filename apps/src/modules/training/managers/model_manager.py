from apps.src.modules.training.model_config.KoBERT_config import KoBERTConfig
from apps.src.models.KoBERT_classifier import KoBERTClassifier


class ModelManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is not None:
            return cls._instance
        cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, config):
        if hasattr(self, "initialized") and self.initialized:
            self.config = config
            if self.config['model_type'] == 'KoBERT':
                config_instance = KoBERTConfig(
                    self.config['model_type'], self.config['base_dir'], self.config['text_dataset']
                )
                config_instance.set_model_config(['model_type'], self.config['model_type'])
                config_instance.set_model_config(['base_dir'], self.config['base_dir'])
                config_instance.set_model_config(['text_dataset'], self.config['text_dataset'])
                config_instance.set_model_config(['gpu_id'], self.config['gpu_id'])
                config_instance.set_model_config(['load_trained_model'], self.config['load_trained_model'])
                config_instance.set_model_config(['load_model_name'], self.config['load_model_name'])
                return

        self.config = config
        self.kobert_config = KoBERTConfig(
            self.config['model_type'], self.config['base_dir'], self.config['text_dataset']
        )
        self.initialized = True

    def initialize_model(self, num_labels):
        if self.config['model_type'] == 'KoBERT':
            return KoBERTClassifier(num_labels=num_labels, train_config=self.config)

        return None

    def update_and_save_config(self):
        if self.config['model_type'] == 'KoBERT':
            updated_config = self.get_updated_kobert_config()
            updated_kobert_config = self.dict_recursive_update(self.kobert_config.model_config, updated_config)
            self.kobert_config.save_model_config(updated_kobert_config)

    def dict_recursive_update(self, orig_dict, new_dict):
        merged_dict = orig_dict.copy()
        for key, val in new_dict.items():
            if key in merged_dict and isinstance(merged_dict[key], dict) and isinstance(val, dict):
                merged_dict[key] = self.dict_recursive_update(merged_dict[key], val)
            else:
                merged_dict[key] = val
        return merged_dict

    def get_updated_kobert_config(self):
        return {
            'model_type': self.config['model_type'],
            'text_dataset': self.config['text_dataset'],
            'base_dir': self.config['base_dir'],
            'gpu_id': self.config['gpu_id'],
            'load_trained_model': self.config['load_trained_model'],
            'load_model_name': self.config['load_model_name'],
            'filename_extension': self.config['filename_extension'],

            'tokenizer': {
                'padding': self.config['tokenizer']['padding'],
            },

            'dataloader': {
                'delimiter': self.config['dataloader']['delimiter'],
                'column_names': self.config['dataloader']['column_names'],
            },

            'trainer_args': {
                'per_device_train_batch_size': self.config['trainer_args']['per_device_train_batch_size'],
                'per_device_eval_batch_size': self.config['trainer_args']['per_device_eval_batch_size'],
                'num_train_epochs': self.config['trainer_args']['num_train_epochs'],
                'evaluation_strategy': self.config['trainer_args']['evaluation_strategy'],
                'steps': self.config['trainer_args']['steps'],
                'learning_rate': self.config['trainer_args']['learning_rate'],
                'optim': self.config['trainer_args']['optim'],
                'warmup_ratio': self.config['trainer_args']['warmup_ratio'],
                'warmup_steps': self.config['trainer_args']['warmup_steps'],
                'lr_scheduler_type': self.config['trainer_args']['lr_scheduler_type'],
                'output_dir': self.config['trainer_args']['output_dir'],
                'logging_dir': self.config['trainer_args']['logging_dir'],
                'weight_decay': self.config['trainer_args']['weight_decay'],
                'early_stopping_patience': self.config['trainer_args']['early_stopping_patience'],
            },

            'KoBERT': {
                'max_length': self.config['KoBERT']['max_length'],
            }
        }
