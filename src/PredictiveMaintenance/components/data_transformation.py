import os
import pandas as pd
import numpy as np
from datetime import datetime
import shutil
from PredictiveMaintenance.logging import logger
from PredictiveMaintenance.utils import common
from PredictiveMaintenance.entity import DataTransformationTrainingConfig


class DataTransformationTraining:
    def __init__(self, config: DataTransformationTrainingConfig) -> None:
        self.config = config

    def replace_missing_values_with_null(self, ):
         """
            This method replaces the missing values in column with "NULL" to 
            store into the table.

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
            logger.info(f""">>>>>>> Replacing missing values with null for trainining files started... <<<<<<<""")

            all_items = os.listdir(self.config.good_dir)  # all items in the directory
            only_files = [item for item in all_items if os.path.isfile(os.path.join(self.config.good_dir, item)) and item != ".DS_Store"]

            for file in only_files:
                df = pd.read_csv(os.path.join(self.config.good_dir, file),
                                sep='\s+', 
                                names=self.config.column_names, 
                                nrows=10)
                df = df.fillna(np.nan)

                df.to_csv(os.path.join(self.config.good_dir, file), index=False, header=True)
            
            logger.info(f""">>>>>>> Replacing missing values with null for trainining files completed. <<<<<<<""")
         
         except Exception as e:
            logger.exception(e)
            raise e
    
    def delete_existing_good_data_training_folder(self, ):
        """
            This method deletes the directory good_raw, after loading the data into the table.
            Once the good files are loaded in DB, deleting the directory ensures space optimization.

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
            logger.info(f""">>>>>>> Attempting to remove existing 'good_raw' directory... <<<<<<<""")
            
            common.remove_directories([self.config.good_dir])

            logger.info(f""">>>>>>> Removing 'good_raw' directory is successful. <<<<<<<""")
        
        except Exception as e:
            logger.exception(e)
            raise e
    
    def delete_exising_bad_data_training_folder(self, ):
        """
            This method deletes the 'bad_raw' directory.

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
            logger.info(f""">>>>>>> Attempting to delete 'bad_raw' directory... <<<<<<<""")

            if os.path.isdir(self.config.bad_dir):
                common.remove_directories([self.config.bad_dir])
                logger.info(f""">>>>>>> 'bad_raw' directory removed successfully... <<<<<<<""")
            else:
                logger.info(f""">>>>>>> 'bad_raw' directory doesn't exists... <<<<<<<""")
        
        except OSError as o:
            logger.exception(f"""Error while deleting directory, Error message : {str(o)}""")
            raise e

        except Exception as e:
            logger.exception(f"""Exception while deleting directory, Exception message : {str(e)}""")
            raise e
        
    
    def move_bad_files_to_archive_bad(self, ):
        """
            This method deletes the directory made to store the Bad Data,
            after moving the data in an archive folder. We archive the bad files to send them
            back to the client for invalid data issue. 

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
            logger.info(f""">>>>>>> Attempting to move bad files to archive bad... <<<<<<<""")
            now = datetime.now()
            date = now.date()
            time = now.strftime("%H:%M:%S")
            
            if os.path.isdir(self.config.bad_dir):
                if not os.path.isdir(self.config.archive_bad_dir):
                    common.create_directories([self.config.archive_bad_dir])
                
                bad_data_dir = "bad_data_" + str(date) + "_" + str(time)
                destination = os.path.join(self.config.archive_bad_dir, bad_data_dir)
                if not os.path.isdir(destination):
                    common.create_directories([destination])
                
                files = os.listdir(self.config.bad_dir)
                for file in files:  # moving bad files to bad arvhive directory
                    if file not in os.listdir(destination):
                        shutil.move(os.path.join(self.config.bad_dir, file), destination)

            logger.info(f""">>>>>>> Bad files {str(files)} are moved to archive directory '{str(destination)}'! <<<<<<<""")
            
            # delete bad_raw directory after copying bad files to bad archive directory
            self.delete_exising_bad_data_training_folder()

        except Exception as e:
            logger.exception(e)
            raise e
        
