import os
import re
import shutil
import pandas as pd

from PredictiveMaintenance.logging import logger
from PredictiveMaintenance.entity import (DataValidationTrainingConfig)


class DataValidationTraining:
    def __init__(self, 
                 config:DataValidationTrainingConfig,) -> None:
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
    
    def training_validate_column_length(self,):
        """
            This method validates the number of columns in the csv file.

            If the number of columns is same as given in the schema, the file is kept inside good_raw,
            otherwise, if there is a mismatch between given number of columns and that is specified
            in the schema, then corresponding file is moved to bad_raw directory

            Parameters
            ----------
            None

            Returns
            -------
            None

            Raises
            ------
            OSError
            Exception
        
        """
        try:
            logger.info(f""">>>>>>> Column length validation of trainining files started... <<<<<<<""")

            number_of_columns = self.config.number_of_columns

            all_items = os.listdir(self.config.good_dir)  # all items in the directory
            only_files = [item for item in all_items if os.path.isfile(os.path.join(self.config.good_dir, item)) and item != ".DS_Store"]
            
            for file in only_files:
                df = pd.read_csv(os.path.join(self.config.good_dir, file),
                                sep='\s+', 
                                header=None, 
                                nrows=10)
                if df.shape[1] == number_of_columns:
                    pass
                else:
                    shutil.move(os.path.join(self.config.good_dir, file), self.config.bad_dir)
                    logger.info(f"""File '{str(file)}' has invalid number of columns. 
                                Therefore moved to 'bad_raw' directory.""")
            
            logger.info(f""">>>>>>> Column length validation of trainining files completed. <<<<<<<""")
        except OSError as o:
            logger.exception(o)
            raise o
        
        except Exception as e:
            logger.exception(e)
            raise e
    
    def training_validate_missing_values_in_whole_column(self, ):
        """
            If any column in the data file has all the values as missing, then 
            such files are not suitable for processing. Therefore, this method moves
            corresponding files to bad_raw directory.

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
            logger.info(f""">>>>>>> Missing values in whole column validation of trainining files started... <<<<<<<""")
            
            all_items = os.listdir(self.config.good_dir)  # all items in the directory
            only_files = [item for item in all_items if os.path.isfile(os.path.join(self.config.good_dir, item)) and item != ".DS_Store"]

            for file in only_files:
                df = pd.read_csv(os.path.join(self.config.good_dir, file),
                                sep='\s+', 
                                header=None, 
                                nrows=10)
                
                all_nan_columns = df.isna().all()
                all_nan_columns_list = list(all_nan_columns[all_nan_columns.values == True].index)
                if len(all_nan_columns_list) > 0:
                    source_path = os.path.join(self.config.good_dir, file)
                    destination_path = self.config.bad_dir
                    shutil.move(source_path, destination_path)
                    logger.info(f""">>>>>>> The file '{str(file)}' contain columns with whole missing values. 
                                Therefore, moved to 'bad_raw' directory. <<<<<<<""")
            
            logger.info(f""">>>>>>> Missing values in whole column validation of trainining files completed. <<<<<<<""")
        
        except OSError as o:
            logger.exception(o)
            raise o
        
        except Exception as e:
            logger.exception(e)
            raise e

