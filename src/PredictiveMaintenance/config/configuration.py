from PredictiveMaintenance.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from PredictiveMaintenance.utils import common
from PredictiveMaintenance.entity import (DataIngestionConfig,
                                          DataValidationTrainingConfig,
                                          DataValidationTrainingParams)

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH) -> None:
        self.config = common.read_yaml(config_filepath)
        self.params = common.read_yaml(params_filepath)

        common.create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        common.create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            destination_folder=config.destination_folder,
            train_file_name_regex=config.train_file_name_regex,
            test_file_name_regex1 = config.test_file_name_regex1,
            test_file_name_regex2 = config.test_file_name_regex2,
            train_destination_folder = config.train_destination_folder,
            test_destination_folder = config.test_destination_folder,
            miscellaneous_folder = config.miscellaneous_folder
        )
    
        return data_ingestion_config
    
    def get_data_validation_training_cofig(self) -> DataValidationTrainingConfig:
        config = self.config.data_validation_training

        common.create_directories([config.good_dir, config.bad_dir])

        data_validation_training_config = DataValidationTrainingConfig(
            root_dir = config.root_dir,
            good_dir = config.good_dir,
            bad_dir = config.bad_dir
        )

        return data_validation_training_config
    
    def get_data_validation_training_params(self) -> DataValidationTrainingParams:
        params = self.params.data_validation_training_params

        data_validation_training_params = DataValidationTrainingParams(
            column_names = params.column_names
        )

        return data_validation_training_params