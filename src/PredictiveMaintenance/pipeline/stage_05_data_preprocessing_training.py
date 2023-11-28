import pandas as pd
from PredictiveMaintenance.config.configuration import ConfigurationManager
from PredictiveMaintenance.components.data_preprocessing_training import DataPreProcessingTraining


class DataPreProcessingTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self,):
        config = ConfigurationManager()
        data_preprocessing_training_config = config.get_data_preprocessing_training_config()

        data_preprocessing_training = DataPreProcessingTraining(config=data_preprocessing_training_config)
        
        # loading data
        df = data_preprocessing_training.load_input_data_for_training()
        
        # adding remaining useful life("RUL") column
        df = data_preprocessing_training.add_remaining_useful_life(df)
        
        # separating features and label
        X, y = data_preprocessing_training.separate_label_feature(df, label_column_name="RUL")
        
        # impuring missing values
        is_null_present = data_preprocessing_training.is_null_present(X)
        if is_null_present:
            X = data_preprocessing_training.impute_missing_values(X)
        
        # dropping columns with Zero standard deviation
        cols_to_drop = data_preprocessing_training.get_columns_with_zero_std_deviation(X)
        if len(cols_to_drop) > 0:
            X = data_preprocessing_training.remove_columns(X, cols_to_drop)
        
        # min-max scaling
        X = data_preprocessing_training.training_min_max_scaling(X)

        # saving pre-processed data for training
        data_preprocessing_training.save_data(X, y)

