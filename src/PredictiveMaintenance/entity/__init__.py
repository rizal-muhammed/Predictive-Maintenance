from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    destination_folder: Path
    train_file_name_regex: str
    test_file_name_regex1: str
    test_file_name_regex2: str
    train_destination_folder: Path
    test_destination_folder: Path
    miscellaneous_folder: Path

@dataclass(frozen=True)
class DataValidationTrainingConfig:
    root_dir: Path
    good_dir: Path
    bad_dir: Path
    training_source_dir: Path
    file_name_regex: str
    number_of_columns: int

@dataclass(frozen=True)
class DataValidationTrainingParams:
    column_names: list

@dataclass(frozen=True)
class DataTransformationTrainingConfig:
    good_dir: Path
    bad_dir: Path
    archive_bad_dir: Path
    column_names: list

@dataclass(frozen=True)
class DataBaseOperationsTrainingConfig:
    root_dir: Path
    file_name: str
    good_dir: Path
    bad_dir: Path

@dataclass(frozen=True)
class DataBaseOperationsTrainingCredentials:
    host: str
    user: str
    password: str

@dataclass(frozen=True)
class DataBaseOperationsTrainingParams:
    db_name: str
    table_name: str
    column_names: dict

@dataclass(frozen=True)
class DataPreProcessingTrainingConfig:
    root_dir: Path
    input_filepath: Path
    preprocessed_input_data_dir: Path

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    models_dir: Path
    figures_dir: Path
    preprocessed_X: Path
    preprocessed_y: Path


@dataclass(frozen=True)
class ModelTrainingParams:
    linear_regression_params: dict
    random_forest_params: dict
    svr_params: dict
    test_size: float