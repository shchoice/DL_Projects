from apps.src.modules.train.model_config.base_config import BaseConfig


class KoBERTConfig(BaseConfig):
    _default_config = {
        'model_type': None,
        'text_dataset': None,
        'base_dir': None,
        'gpu_id': 'cpu',
        'load_trained_model': None,
        'load_model_name': None,
        'filename_extension': '.tsv',

        'tokenizer': {
            'padding': 'max_length',
        },

        'dataloader': {
            'delimiter': '\t',
            'column_names': ['ID', 'Category', 'Text'],
        },

        'trainer_args': {
            'per_device_train_batch_size': 64,
            'per_device_eval_batch_size': 32,
            'num_train_epochs': 2,
            'evaluation_strategy': 'steps',
            'steps': 100,
            'learning_rate': 5e-4,
            'optim': 'adamw_torch',
            'warmup_ratio': 0,
            'warmup_steps': 0,
            'lr_scheduler_type': 'cosine',
            'output_dir': None,
            'logging_dir': None,
            'weight_decay': 0.1,
            'early_stopping_patience': 10,
        },

        'KoBERT': {
            'max_length': 64,
        }
    }
