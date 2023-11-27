import os
import requests
import zipfile
from io import BytesIO
import re
import shutil

from PredictiveMaintenance.entity import DataIngestionConfig
from PredictiveMaintenance.logging import logger
from PredictiveMaintenance.utils import common


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
    
    def prepare_files(self):
        try:
            common.create_directories([self.config.train_destination_folder,
                                    self.config.test_destination_folder,
                                    self.config.miscellaneous_folder])

            all_items = os.listdir(self.config.root_dir)
            only_files = [item for item in all_items if os.path.isfile(os.path.join(self.config.root_dir, item)) and item != ".DS_Store"]

            for file in only_files:
                if re.match(self.config.train_file_name_regex, file):
                    shutil.move(os.path.join(self.config.root_dir, file),
                                self.config.train_destination_folder)
                elif re.match(self.config.test_file_name_regex1, file):
                    shutil.move(os.path.join(self.config.root_dir, file),
                                self.config.test_destination_folder)
                elif re.match(self.config.test_file_name_regex2, file):
                    shutil.move(os.path.join(self.config.root_dir, file),
                                self.config.test_destination_folder)
                else:
                    shutil.move(os.path.join(self.config.root_dir, file),
                                self.config.miscellaneous_folder)
        except Exception as e:
            raise e