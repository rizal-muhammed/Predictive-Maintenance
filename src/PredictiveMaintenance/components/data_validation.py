import os
import re
import shutil

from PredictiveMaintenance.entity import DataValidationTrainingConfig
from PredictiveMaintenance.logging import logger


class DataValidationTraining:
    def __init__(self, config:DataValidationTrainingConfig) -> None:
        self.config = config
    
    def training_raw_file_name_validation(self,):
        """
            This function validates the name of the training csv files as per 
            given naming convention specified in the schema.

            pre-defined regex pattern is used to validate the file name. 
            If name format does not match, the file is moved to 'bad_raw' data folder 
            If name format matches, then the file is moved to 'good_raw' Data folder 

            Parameters
            ----------
            None

            Returns
            -------
            None

            Raises
            ------
            Exception
        
        """
        try:
            logger.info(f""">>>>>>> File name validation of trainining files started... <<<<<<<""")

            all_items = os.listdir(self.config.training_source_dir)  # all items in the directory
            only_files = [item for item in all_items if os.path.isfile(os.path.join(self.config.training_source_dir, item)) and item != ".DS_Store"]

            for file in only_files:
                if re.match(self.config.file_name_regex, file):
                    shutil.copy(os.path.join(self.config.training_source_dir, file),
                                self.config.good_dir)
                    logger.info(f"""File {str(file)} moved to 'good_raw' directory.""")
                else:
                    shutil.copy(os.path.join(self.config.training_source_dir, file),
                                self.config.bad_dir)
                    logger.info(f"""File {str(file)} moved to 'bad_raw' directory.""")
            
            logger.info(f""">>>>>>> File name validation of trainining files completed. <<<<<<<""")
        except Exception as e:
            logger.exception(f"""Exception during file name validation of training files. 
                             Exception message : {str(e)}""")
            raise e
                