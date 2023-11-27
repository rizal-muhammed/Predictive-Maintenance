from PredictiveMaintenance.config.configuration import ConfigurationManager
from PredictiveMaintenance.components.data_ingestion import DataIngestion
from PredictiveMaintenance.logging import logger


class DataIngestionPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_and_extract_file()
        data_ingestion.prepare_files()