from apps.src.config.model_config.KoBERT_config import KoBERTConfig
from apps.src.models.KoBERT_classifier import KoBERTClassifier


class ModelManager:
    def __init__(self, train_config):
        self.train_config = train_config
        self.kobert_config = KoBERTConfig(
            self.train_config['model_type'], self.train_config['base_dir'], self.train_config['text_dataset']
        )

    def initialize_model(self, num_labels):
        if self.train_config['model_type'] == 'KoBERT':
            return KoBERTClassifier(num_labels=num_labels, train_config=self.train_config)

        return None

    def update_and_save_config(self):
        if self.train_config['model_type'] == 'KoBERT':
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
            'model_type': self.train_config['model_type'],
            'text_dataset': self.train_config['text_dataset'],
            'base_dir': self.train_config['base_dir'],
            'gpu_id': self.train_config['gpu_id'],
            'load_trained_model': self.train_config['load_trained_model'],
            'load_model_name': self.train_config['load_model_name'],
            'filename_extension': self.train_config['filename_extension'],

            'tokenizer': {
                'padding': self.train_config['tokenizer']['padding'],
            },

            'dataloader': {
                'delimiter': self.train_config['dataloader']['delimiter'],
                'column_names': self.train_config['dataloader']['column_names'],
            },

            'trainer_args': {
                'per_device_train_batch_size': self.train_config['trainer_args']['per_device_train_batch_size'],
                'per_device_eval_batch_size': self.train_config['trainer_args']['per_device_eval_batch_size'],
                'num_train_epochs': self.train_config['trainer_args']['num_train_epochs'],
                'evaluation_strategy': self.train_config['trainer_args']['evaluation_strategy'],
                'steps': self.train_config['trainer_args']['steps'],
                'learning_rate': self.train_config['trainer_args']['learning_rate'],
                'optim': self.train_config['trainer_args']['optim'],
                'warmup_ratio': self.train_config['trainer_args']['warmup_ratio'],
                'warmup_steps': self.train_config['trainer_args']['warmup_steps'],
                'lr_scheduler_type': self.train_config['trainer_args']['lr_scheduler_type'],
                'output_dir': self.train_config['trainer_args']['output_dir'],
                'logging_dir': self.train_config['trainer_args']['logging_dir'],
                'weight_decay': self.train_config['trainer_args']['weight_decay'],
                'early_stopping_patience': self.train_config['trainer_args']['early_stopping_patience'],
            },

            'KoBERT': {
                'max_length': self.train_config['KoBERT']['max_length'],
            }
        }