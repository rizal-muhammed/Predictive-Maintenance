from PredictiveMaintenance.config.configuration import ConfigurationManager
from PredictiveMaintenance.components.model_training import ModelTraining


class ModelTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self, ):
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_training_params = config.get_model_training_params()

        model_training = ModelTraining(config=model_training_config,
                                       params=model_training_params,)
        X, y = model_training.load_data()
        
        X_train, X_val, y_train, y_val = model_training.train_test_split(X, y)

        model_training.train(X_train, X_val, y_train, y_val)
        


