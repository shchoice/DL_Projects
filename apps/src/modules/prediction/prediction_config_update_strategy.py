from apps.src.modules.common.config_update_strategy import ConfigUpdateStrategy


class PredictionConfigUpdateStrategy(ConfigUpdateStrategy):
    def update_configuration(self, config, schema):
        config['top_k'] = schema.top_k
        config['documents'] = schema.documents
        config['gpu_id'] = schema.gpu_id
        config['load_trained_model'] = schema.load_trained_model
        config['load_model_name'] = schema.load_model_name
