from apps.src.modules.common.config_update_strategy import ConfigUpdateStrategy


class TrainingConfigUpdateStrategy(ConfigUpdateStrategy):
    def update_configuration(self, config, schema):
        config['gpu_id'] = schema.gpu_id
        config['load_trained_model'] = schema.load_trained_model
        config['load_model_name'] = schema.load_model_name
