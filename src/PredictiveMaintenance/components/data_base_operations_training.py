import os
import shutil
import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import errorcode
from sqlalchemy import create_engine
from pathlib import Path

from PredictiveMaintenance.logging import logger
from PredictiveMaintenance.utils import common
from PredictiveMaintenance.entity import (DataBaseOperationsTrainingConfig, 
                                          DataBaseOperationsTrainingCredentials,
                                          DataBaseOperationsTrainingParams)


class DataBaseOperations:
    def __init__(self,
                 config:DataBaseOperationsTrainingConfig,
                 credentials:DataBaseOperationsTrainingCredentials,
                 params:DataBaseOperationsTrainingParams) -> None:
        self.config = config
        self.credentials = credentials
        self.params=params

        common.create_directories([Path(self.config.root_dir)])

    def database_connection_establishment(self, ):
        """
            This method creates database(if not exists) and establishes the connection
            to the database

            Parameters
            ----------
            None

            Returns
            -------
            Returns the connection to the corresponding database specified in the params.

            Raises
            ------
            ConnectionError
            Exception
        
        """
        try:
            logger.info(f""">>>>>>> attempting db connection establishment for trainining files... <<<<<<<""")

            conx = mysql.connector.connect(host=self.credentials.host,
                                       user=self.credentials.user,
                                       password=self.credentials.password)
            mycursor = conx.cursor()
            query = f"create database if not exists {self.params.db_name}"
            mycursor.execute(query)

            logger.info(f""">>>>>>> The connection establishment is successful, and database is created successfully if not exists. <<<<<<<""")

            return conx


        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                logger.exception(f""">>>>>>> Something is wrong with the user name or password <<<<<<<""")
            else:
                logger.exception(f""">>>>>>>Connection Error occured : {err} <<<<<<<""")
            raise err

        except Exception as e:
            logger.exception(f""">>>>>>>Database creation is failed or database connection establishment is failed : {e} <<<<<<<""")
            raise e
        
        finally:
            if mycursor is not None:
                mycursor.close()

    
    def create_table_db(self, ):
        """
            This method creates table specified in the params (if not exists) in the given database, 
            which  will be used to insert the good data after raw data validation

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
            logger.info(f""">>>>>>> creating table for trainining files... <<<<<<<""")

            conx = self.database_connection_establishment()
            mycursor = conx.cursor()

            try:
                mycursor.execute("use {}".format(self.params.db_name))
            except Exception as e:
                logger.exception(f"""Couldn't use the database {self.params.db_name} successfully""")
                raise e
        
            # Check if the table exists
            mycursor.execute(f"SHOW TABLES LIKE '{self.params.table_name}'")
            result = mycursor.fetchone()
            if result:
                    logger.info(f"""Table '{str(self.params.table_name)}' already exists.""")
            else:
                for key in self.params.column_names.keys():
                    column_name = str(key)
                    dtype = str(self.params.column_names[key])

                    try:
                        # if table already created, then add the
                        alter_table_query = f"""ALTER TABLE {self.params.table_name} ADD COLUMN `{column_name}` {dtype}"""
                        mycursor.execute(alter_table_query)
                    except Exception as e:
                        # table doesn't exists. so create table
                        create_table_query = f"""
                        CREATE TABLE IF NOT EXISTS {self.params.table_name} (
                            `{column_name}` {dtype}
                        )
                        """
                        mycursor.execute(create_table_query)
                        logger.info(f"""table {self.params.table_name} created successfully""")
            conx.commit()
        
        except Exception as e:
            logger.exception(f"""Exception while creating the table {str(self.params.table_name)}.
                             Exception message : {str(e)}""")
            raise e
        
        finally:
            if mycursor is not None:
                mycursor.close()

            if conx.is_connected():
                conx.close()
    
    def insert_into_table_good_data(self, ):
        """
            This method inserts the data from good_raw files to the given database table

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
            logger.info(f""">>>>>>> insertion of good data into database table for trainining files started... <<<<<<<""")

            conx = self.database_connection_establishment()
            mycursor = conx.cursor()
            
            try:
                mycursor.execute("use {}".format(self.params.db_name))
            except Exception as e:
                logger.exception(f"""Couldn't use the database {self.params.db_name} successfully""")
                raise e
            
            all_items = os.listdir(self.config.good_dir)  # all items in the good_raw directory
            # listing only the files in the good_raw directory
            only_files = [item for item in all_items if os.path.isfile(os.path.join(self.config.good_dir, item)) and item != ".DS_Store"]

            engine = create_engine(f"""mysql+mysqlconnector://{self.credentials.user}:{self.credentials.password}@{self.credentials.host}/{self.params.db_name}""")

            # opening good data files one be one and inserting the data into the given 
            for file in only_files:
                try:
                    file_path = os.path.join(self.config.good_dir, file)
                    df = pd.read_csv(file_path)
                    df = df.fillna(np.nan)

                    if(list(self.params.column_names.keys()) == list(df.columns)):
                        df.to_sql(name=self.params.table_name, con=engine, if_exists="append", index=False)
                        conx.commit()
                    else:
                        logger.error(f"Could not insert into the table {str(self.params.table_name)}")
                        shutil.move(os.path.join(self.config.good_dir, file), self.config.bad_dir)
                        logger.info(f"""Since insertion failed, the file '{str(file)}' moved to 'bad_raw' data successfully""")

                except Exception as e:
                    logger.exception( f"Exception occured while inserting data into the table : {str(e)}")
                    shutil.move(os.path.join(self.config.good_dir, file), self.config.bad_dir)
                    raise e
            
            logger.info(f"All the 'good_raw' files are inserted successfully into the 'good_raw' table")

        except Exception as e:
            conx.rollback()
            logger.exception( f"Exception occured while inserting data into the table : {str(e)}")
            raise e
        
        finally:
            if mycursor is not None:
                mycursor.close()

            if conx.is_connected():
                conx.close()

            engine.dispose()
    
    def export_data_from_table_into_csv(self, ):
        """
            This method exports from the database into a csv file

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
            logger.info(f""">>>>>>> exporting good data into csv file from training database... <<<<<<<""")

            conx = self.database_connection_establishment()
            mycursor = conx.cursor()

            try:
                mycursor.execute("use {}".format(self.params.db_name))
            except Exception as e:
                logger.exception(f"""Couldn't use the database {self.params.db_name} successfully""")
                raise e
            
            query = f"select * from {self.params.table_name}"  # select query
            mycursor.execute(query)

            data = mycursor.fetchall()  # fetching all the data
            column_names = [desc[0] for desc in mycursor.description]  # extracting column names from description

            df = pd.DataFrame(data, columns=column_names)  # creating a pandas dataframe of data

            if not os.path.isdir(self.config.root_dir):  # creating a directory(if not exists) to store the data fetched from the database
                common.create_directories([self.config.root_dir])
            filepath = os.path.join(self.config.root_dir, self.config.file_name)

            df.to_csv(filepath, index=False, header=True)  # exporting to csv file

            logger.info(f"Input data exported to csv successfully")
        
        except Exception as e:
            logger.exception(f"Input data exporting to csv is failed, Error : {str(e)}")
            conx.close()
            raise e

        finally:
            if mycursor is not None:
                mycursor.close()
                
            if conx.is_connected():
                conx.close()



