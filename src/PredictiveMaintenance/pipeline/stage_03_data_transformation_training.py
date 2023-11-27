from PredictiveMaintenance.config.configuration import ConfigurationManager
from PredictiveMaintenance.components.data_transformation import DataTransformationTraining


class DataTransformationTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_training_config = config.get_data_transformation_training_config()
        data_transformation_training = DataTransformationTraining(config=data_transformation_training_config)
        data_transformation_training.replace_missing_values_with_null()