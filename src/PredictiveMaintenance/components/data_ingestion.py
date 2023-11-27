import os
import requests
import zipfile
from io import BytesIO

from PredictiveMaintenance.entity import DataIngestionConfig
from PredictiveMaintenance.logging import logger


class DataIngestion:
    def __init__(self, config:DataIngestionConfig) -> None:
        self.config = config

    def download_and_extract_file(self):
        response = requests.get(self.config.source_URL)

        if response.status_code == 200:
            with zipfile.ZipFile(BytesIO(response.content), 'r') as zip_ref:
                zip_ref.extractall(self.config.destination_folder)
            logger.info(f"""Data successfully downloaded and extracted to {self.config.destination_folder}""")
        else:
            logger.error(f"""Failed to download file. Status code: {response.status_code}""")
