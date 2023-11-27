from PredictiveMaintenance.config.configuration import ConfigurationManager
from PredictiveMaintenance.components.data_validation import DataValidationTraining
from PredictiveMaintenance.logging import logger

class DataValidationTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_training_config = config.get_data_validation_training_config()
        data_validation_training = DataValidationTraining(config=data_validation_training_config)
        data_validation_training.training_raw_file_name_validation()
        data_validation_training.training_validate_column_length()
        data_validation_training.training_validate_missing_values_in_whole_column()
        