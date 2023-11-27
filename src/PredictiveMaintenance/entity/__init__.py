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

@dataclass(frozen=True)
class DataValidationTrainingParams:
    column_names: list