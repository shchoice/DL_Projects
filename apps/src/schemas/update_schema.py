from typing import Optional

from pydantic import BaseModel, Field


class UpdateSchema(BaseModel):
    model_type: str
    base_dir: str
    collection: str
    gpu_id: Optional[str] = Field(default='auto', config_path=None)
    load_trained_model: Optional[bool] = Field(default='false', config_path=None)
    load_model_name: Optional[str] = Field(default='checkpoint-best', config_path=None)
    filename_extension: Optional[str] = Field(default='.tsv', config_path=None)

    padding: Optional[str] = Field(default='max_length', config_path=['tokenizer', 'padding'])

    delimiter: Optional[str] = Field(default='\t', config_path=['dataloader', 'delimiter'])
    column_names: Optional[list] = Field(default=['ID', 'Category', 'Text'], config_path=['dataloader', 'column_names'])

    per_device_train_batch_size: Optional[int] = Field(default=64, config_path=['trainer_args', 'per_device_train_batch_size'])
    per_device_eval_batch_size: Optional[int] = Field(default=32, config_path=['trainer_args', 'per_device_eval_batch_size'])
    num_train_epochs: Optional[int] = Field(default=10, config_path=['trainer_args', 'num_train_epochs'])
    evaluation_strategy: Optional[str] = Field(default='steps', config_path=['trainer_args', 'evaluation_strategy'])
    steps: Optional[int] = Field(default=100, config_path=['trainer_args', 'steps'])
    learning_rate: Optional[float] = Field(default=5e-4, config_path=['trainer_args', 'learning_rate'])
    optim: Optional[str] = Field(default='adamw_torch', config_path=['trainer_args', 'optim'])
    warmup_ratio: Optional[float] = Field(default=0.1, config_path=['trainer_args', 'warmup_ratio'])
    warmup_steps: Optional[int] = Field(default=0, config_path=['trainer_args', 'warmup_steps'])
    lr_scheduler_type: Optional[str] = Field(default='cosine', config_path=['trainer_args', 'lr_scheduler_type'])
    output_dir: Optional[str] = Field(default=None, config_path=['trainer_args', 'output_dir'])
    logging_dir: Optional[str] = Field(default=None, config_path=['trainer_args', 'logging_dir'])
    weight_decay: Optional[float] = Field(default=0.1, config_path=['trainer_args', 'weight_decay'])
    early_stopping_patience: Optional[int] = Field(default=10, config_path=['trainer_args', 'early_stopping_patience'])

    max_length: Optional[int] = Field(default=32)

    def get_config_path(self, field_name):
        if field_name == "max_length" and self.model_type:
            return [self.model_type, 'max_length']
        else:
            field_info = self.__fields__[field_name].field_info
            return getattr(field_info, 'config_path', [field_name])

    def extract_config(self):
        config_data = {}
        for field_name, field_value in self:
            if field_value is not None:
                key_path = self.get_config_path(field_name)
                self.set_nested_dict(config_data, key_path, field_value)
        return config_data

    @staticmethod
    def set_nested_dict(data_dict, key_path, value):
        for key in key_path[:-1]:
            data_dict = data_dict.setdefault(key, {})
        data_dict[key_path[-1]] = value
