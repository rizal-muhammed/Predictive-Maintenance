from PredictiveMaintenance.entity import (DataIngestionConfig)
from PredictiveMaintenance.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from PredictiveMaintenance.utils import common

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
            destination_folder=config.destination_folder
        )
    
        return data_ingestion_config