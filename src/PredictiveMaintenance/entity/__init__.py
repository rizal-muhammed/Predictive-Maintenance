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