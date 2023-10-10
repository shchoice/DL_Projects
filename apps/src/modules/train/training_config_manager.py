import os

from transformers import TrainingArguments

from apps.src.config import constants


class TrainingConfigManager:
    def __init__(self, train_config):
        self.train_config = train_config

    def get_training_arguments(self):
        base_output_dir = os.path.join(self.train_config['base_dir'], constants.MODEL_OUTPUT_PATH_NAME)
        dataset_name = self.train_config['text_dataset']

        args = TrainingArguments(
            per_device_train_batch_size=self.train_config['trainer_args']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.train_config['trainer_args']['per_device_eval_batch_size'],
            num_train_epochs=self.train_config['trainer_args']['num_train_epochs'],
            evaluation_strategy=self.train_config['trainer_args']['evaluation_strategy'],
            eval_steps=self.train_config['trainer_args']['steps'],
            logging_steps=self.train_config['trainer_args']['steps'],
            learning_rate=float(self.train_config['trainer_args']['learning_rate']),
            save_strategy=self.train_config['trainer_args']['evaluation_strategy'],
            save_steps=self.train_config['trainer_args']['steps'],
            output_dir=os.path.join(base_output_dir, f'output-{dataset_name}'),
            logging_dir=os.path.join(base_output_dir, f'log-{dataset_name}'),
            fp16=True,
            load_best_model_at_end=True,
            weight_decay=self.train_config['trainer_args']['weight_decay'],
            lr_scheduler_type='cosine',
        )

        return args