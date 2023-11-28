import pandas as pd
import numpy as np
import os
import pickle
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

from ensure import ensure_annotations
from PredictiveMaintenance.logging import logger
from PredictiveMaintenance.utils import common
from PredictiveMaintenance.entity import (DataPreProcessingTrainingConfig)



class DataPreProcessingTraining:
    def __init__(self, 
                 config:DataPreProcessingTrainingConfig,) -> None:
        self.config = config

        common.create_directories([self.config.root_dir])
    
    def load_input_data_for_training(self, ):
        """
            This method loads input data for training and returns the data as pandas DataFrame type.

            Parameters
            ----------
            None

            Returns
            -------
            df: Pandas DataFrame type
                Input training data as a Pandas DataFrame.

            Raises
            ------
            Exception
        
        """
        try:
            logger.info(f""" Loading data exported from training database...""")

            df = pd.read_csv(os.path.join(self.config.input_filepath))

            logger.info(f""" Data loaded successfully.""")

            return df
        
        except Exception as e:
            logger.exception(f""" Exception while loading data.""")
            raise e


    
    @ensure_annotations
    def remove_columns(self, df:pd.DataFrame, colums_to_remove:list):
        """
            This method removes the given list of columns from a pandas DataFrame.

            Parameters
            ----------
            df: Pandas DataFrame type
                DataFrame to remove columns
            columns_to_remove: list
                List of columns to remove

            Returns
            -------
            df: Pandas DataFrame type
                Pandas DataFrame after removing the specified list of columns

            Raises
            ------
            Exception
        
        """
        try:
            logger.info(f""" Attempting to remove specified columns from DataFrame... """)

            remaining_df = df.drop(columns=colums_to_remove, axis=1)

            logger.info(f""" Columns {str(colums_to_remove)} successfully removed from DataFrame.""")

            return remaining_df
        
        except Exception as e:
            logger.exception(f"""Exception while removing columns from DataFrame. 
                             Exception message: {str(e)}""")
            raise e
    
    @ensure_annotations
    def add_remaining_useful_life(self, df: pd.DataFrame):
        """
            This method calculates Remaining Useful Life and add as a column.

            Parameters
            ----------
            df : pandas dataframe type
                input files in the form of dataframe

            Returns
            -------
            df : pandas dataframe
                Returns a dataframe with additional column "RUL", which stands for 
                'remaining useful life'

            Raises
            ------
            Exception
        
        """
        train_grouped_by_unit = df.groupby(by='unit_number') 
        max_time_cycles = train_grouped_by_unit['time_cycles'].max() 
        merged = df.merge(max_time_cycles.to_frame(name='max_time_cycle'), left_on='unit_number',right_index=True)
        merged["RUL"] = merged["max_time_cycle"] - merged['time_cycles']
        merged = merged.drop("max_time_cycle", axis=1) 
        return merged

    @ensure_annotations
    def separate_label_feature(self, df: pd.DataFrame, label_column_name: str):
        """
            This method separates the features and label columns.

            Parameters
            ----------
            df : pandas dataframe type
                input data in the form of dataframe
            label_column_name : str type
                label column name to separate from the input data.

            Returns
            -------
            X : pandas DataFrame type
                Returns a DataFrame X of features
            y : pandas DataFrame type
                Returns a DataFrame y of labels

            Raises
            ------
            Exception
        
        """
        try:
            logger.info(f"""Attempting to separate label from features...""")

            X = df.drop(columns=[label_column_name], axis=1)
            y = df[label_column_name].to_frame()

            logger.info(f"""Label separation is successful.""")

            return X, y

        except Exception as e:
            logger.exception(f"""Exception while separating labels from features.""")
            raise e
    
    @ensure_annotations
    def is_null_present(self, df:pd.DataFrame):
        """
            This method checks whether there are null values present in the input dataframe.

            Parameters
            ----------
            df : pandas dataframe type
                input data in the form of dataframe

            Returns
            -------
            null_present : bool type
                True if null values are present in df, False if null values are not present in df.

            Raises
            ------
            Exception

            Notes
            ------
            Saves null count information in 'preprocessed_data/null_value_counts.csv' directory
            for further reference
        
        """
        try:
            logger.info(f"""Attempting to quantify null values in the input data for training...""")

            null_present = False
            null_counts = df.isna().sum()

            for i in null_counts:
                if i > 0:
                    null_present = True
                    break
            
            if null_present:
                df_null_counts = null_counts.to_frame(name="null_count")
                filename = "null_value_counts.csv"
                df_null_counts.to_csv(os.path.join(self.config.root_dir, filename), index=False, header=True)
                logger.info(f"""The null values in the input dataframe is quanitfied at '{str(os.path.join(self.config.root_dir, filename))}',if present.""")

            logger.info(f"""Quantifying null values in the input data for training is completed""")
            
            return null_present
        
        except Exception as e:
            logger.exception(f"""Exception while counting null values in input file.""")
            raise e
    
    @ensure_annotations
    def impute_missing_values(self, df:pd.DataFrame):
        """
            This method replaces all the missing values in the input dataframe using KNNImputer.

            Parameters
            ----------
            df : pandas dataframe type
                input data in the form of a pandas DataFrame

            Returns
            -------
            df_new : pandas DataFrame type
                Returns a dataframe with all the missing values imputed.

            Raises
            ------
            Exception

            Notes
            ------
            The KNN imputer information is stored in 'preprocessed_data/knn_imputer.pkl' directory
            for further reference during prediction.
        
        """
        try:
            logger.info(f"""Impuring missing values from input data...""")

            knn_imputer = KNNImputer(weights="uniform", missing_values=np.nan)
            df_new = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)

            # save model for further reference
            with open(os.path.join(self.config.root_dir, "knn_imputer.pkl"), "wb") as file:
                pickle.dump(knn_imputer, file)

            logger.info(f"""Impute missing values is successful.""")

            return df_new
        
        except Exception as e:
            logger.exception(f"""Exception while imputing missing values in input file.""")
            raise e
    
    @ensure_annotations
    def get_columns_with_zero_std_deviation(self, df:pd.DataFrame):
        """
            This method retuns a list of columsn which have zero standard deviation.


            If the standard deviation is zero, then the column is populated by one value. 
            So if your goal is to prepare the data for regression or classfication, 
            you can throw the column out, since it will contribute nothing to the regression 
            or classification.

            Parameters
            ----------
            df : pandas DataFrame type
                input data in the form of a pandas DataFrame

            Returns
            -------
            coulumns_lst_with_zero_std_dev : list
                Returns a list of column names for which standard deviation is zero.

            Raises
            ------
            Exception

            Notes
            ------
            The list of columns with zero standard deviation is stored in 
            'preprocessed_data/columns_lst_with_zero_std_dev.pkl' directory for further 
            reference during prediction.
        
        """
        try:
            logger.info(f"""Listing columns with zero standard deviation in input file started...""")

            columns = df.columns
            df_description = df.describe()

            columns_with_zero_std_dev = []

            for col in columns:
                if df_description[col]['std'] == 0:
                    columns_with_zero_std_dev.append(col)
            
            if len(columns_with_zero_std_dev) > 0:
                logger.info(f"""Columns with Zero standard deviation are {str(columns_with_zero_std_dev)}""")
            else:
                logger.info(f""" There are no columns with Zero standard deviation.""")
            
            # saving the list for further reference
            with open(os.path.join(self.config.root_dir, "columns_with_zero_std_dev.pkl"), "wb") as file:
                pickle.dump(columns_with_zero_std_dev, file)
            
            logger.info(f"""Listing columns with zero standard deviation in input file is successful.""")

            return columns_with_zero_std_dev
        
        except Exception as e:
            logger.exception(f"""Exception while listing columns with Zero std deviation in input file.""")
            raise e
    
    @ensure_annotations
    def training_min_max_scaling(self, X:pd.DataFrame):
        """
            This method shall be used for performing standard scaling on training data.

            Parameters
            ----------
            X : pandas DataFrame type
                input data in the form of a pandas DataFrame

            Returns
            -------
            X_scaled : pandas DataFrame type
                Returns a X_scaled which is scaled input.

            Raises
            ------
            Exception

            Notes
            ------
            The min_max scaler is stored in 'preprocessed_data/min_max_scaler.pkl' directory 
            for further reference during prediction.
        
        """
        try:
            logger.info(f"""Min Max Scaling of input data started...""")

            min_max_scaler = MinMaxScaler()
            X_scaled = pd.DataFrame(min_max_scaler.fit_transform(X), columns=X.columns)
            print(X_scaled.head())

            # save model for further reference
            with open(os.path.join(self.config.root_dir, "min_max_scaler.pkl"), "wb") as file:
                pickle.dump(min_max_scaler, file)
            
            logger.info(f""" Min-Max scaling of input training data is successful. """)

            return X_scaled

        except Exception as e:
            logger.exception(f"""Exception while scaling of input data.""")
            raise e
    
    @ensure_annotations
    def save_data(self, X:pd.DataFrame, y:pd.DataFrame):
        """
            This method saves the data at the end of pre-processing step at specified directory,
            so that we can retrieve them for model training.

            Parameters
            ----------
            X : pandas DataFrame type
                input features in the form of a pandas DataFrame
            y : pandas DataFrame type
                Ground truth for training

            Returns
            -------
            None

            Raises
            ------
            Exception

            Notes
            ------
            The preprocessed input data is stored at 'artifacts/preprocessed_data/preprocessed_input.csv'
            directory for further reference during prediction.
        
        """
        try:
            logger.info(f"Attempting saving pre-processed data into 'artifacts/preprocessed_data/preprocessed_input.csv' directory...")

            common.create_directories([self.config.preprocessed_input_data_dir])
            X.to_csv(os.path.join(self.config.preprocessed_input_data_dir, "preprocessed_input_X.csv"),
                      index=False,
                      header=True)
            y.to_csv(os.path.join(self.config.preprocessed_input_data_dir, "preprocessed_input_y.csv"),
                      index=False,
                      header=True)
            
            logger.info(f"""Saving pre-processed data is successful.""")

        except Exception as e:
            logger.exception(f"""Exception while saving pre-processed input data.""")
            raise e