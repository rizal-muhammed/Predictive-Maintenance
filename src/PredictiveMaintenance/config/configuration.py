from pathlib import Path
from PredictiveMaintenance.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SECRETS_FILE_PATH
from PredictiveMaintenance.utils import common
from PredictiveMaintenance.entity import (DataIngestionConfig,
                                          DataValidationTrainingConfig,
                                          DataValidationTrainingParams,
                                          DataTransformationTrainingConfig,
                                          DataBaseOperationsTrainingConfig,
                                          DataBaseOperationsTrainingCredentials,
                                          DataBaseOperationsTrainingParams,
                                          DataPreProcessingTrainingConfig,
                                          ModelTrainingConfig,
                                          ModelTrainingParams)

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH,
                 secrets_filepath=SECRETS_FILE_PATH) -> None:
        self.config = common.read_yaml(config_filepath)
        self.params = common.read_yaml(params_filepath)
        self.credentials = common.read_yaml(secrets_filepath)

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
    
    def get_data_validation_training_config(self) -> DataValidationTrainingConfig:
        config = self.config.data_validation_training

        common.create_directories([config.good_dir, config.bad_dir])

        data_validation_training_config = DataValidationTrainingConfig(
            root_dir = config.root_dir,
            good_dir = config.good_dir,
            bad_dir = config.bad_dir,
            training_source_dir = config.training_source_dir,
            file_name_regex = config.file_name_regex,
            number_of_columns = config.number_of_columns,
        )

        return data_validation_training_config
    
    def get_data_validation_training_params(self) -> DataValidationTrainingParams:
        params = self.params.data_validation_training_params

        data_validation_training_params = DataValidationTrainingParams(
            column_names = params.column_names
        )

        return data_validation_training_params
    
    def get_data_transformation_training_config(self) -> DataTransformationTrainingConfig:
        config = self.config.data_transformation_training

        common.create_directories([config.archive_bad_dir])

        data_transformation_training_config = DataTransformationTrainingConfig(
            good_dir = config.good_dir,
            bad_dir = config.bad_dir,
            archive_bad_dir = config.archive_bad_dir,
            column_names = config.column_names,
        )

        return data_transformation_training_config

    def get_data_base_operations_trainig_config(self, ) -> DataBaseOperationsTrainingConfig:
        config = self.config.database_operations_training

        data_base_operations_training_config = DataBaseOperationsTrainingConfig(
            root_dir = config.root_dir,
            file_name = config.file_name,
            good_dir = config.good_dir,
            bad_dir = config.bad_dir,
        )

        return data_base_operations_training_config

    def get_data_base_operations_training_credentials(self, ) -> DataBaseOperationsTrainingCredentials:
        credentials = self.credentials.database_credentials

        data_base_operations_training_credentials = DataBaseOperationsTrainingCredentials(
            host = credentials.host,
            user = credentials.user,
            password = credentials.password,
        )

        return data_base_operations_training_credentials
    
    def get_data_base_operations_training_params(self, ) -> DataBaseOperationsTrainingParams:
        db_params = self.params.database_insertion_training_params

        data_base_operations_training_params = DataBaseOperationsTrainingParams(
            db_name = db_params.db_name,
            table_name = db_params.table_name,
            column_names = db_params.column_names
        )

        return data_base_operations_training_params
    
    def get_data_preprocessing_training_config(self, ) -> DataPreProcessingTrainingConfig:
        data_preprocessing = self.config.data_preprocessing_training
        
        data_preprocessing_training_config = DataPreProcessingTrainingConfig(
            root_dir = data_preprocessing.root_dir,
            input_filepath = data_preprocessing.input_filepath,
            preprocessed_input_data_dir = data_preprocessing.preprocessed_input_data_dir,
        )

        return data_preprocessing_training_config
    
    def get_model_training_config(self, ) -> ModelTrainingConfig:
        model_training = self.config.model_training

        model_training_config = ModelTrainingConfig(
            root_dir = model_training.root_dir,
            models_dir = model_training.models_dir,
            figures_dir = model_training.figures_dir,
            preprocessed_X = Path(model_training.preprocessed_X),
            preprocessed_y = Path(model_training.preprocessed_y)
        )

        return model_training_config
    
    def get_model_training_params(self, ) -> ModelTrainingParams:
        model_training = self.params.model_training_params

        model_training_params = ModelTrainingParams(
            linear_regression_params = model_training.linear_regression_params,
            random_forest_params = model_training.random_forest_params,
            svr_params = model_training.svr_params,
            test_size = model_training.test_size,
        )

        return model_training_params