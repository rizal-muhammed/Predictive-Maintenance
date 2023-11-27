import os
import yaml

from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from pathlib import Path

from PredictiveMaintenance.logging import logger


@ensure_annotations
def read_yaml(path_to_yaml) -> ConfigBox:
    """
        Reads yaml file and returns.

        Parameters
        ----------
        path_to_yaml : str or path-like
            Path to the yaml file.

        Returns
        -------
        ConfigBox or None
            Returns the content of yaml file read in ConfigBox type.

        Raises
        ------
        BoxValueError
            If the yaml file is empty or not in right format.
        Exception

    """
    try:
        path_to_yaml = Path(path_to_yaml)
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
        
    except BoxValueError:
        raise ValueError("yaml file is empty or not in right format.")
    except Exception as e:
        raise e
    
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
        Create directories(if not exists) in the list provided.

        Parameters
        ----------
        path_to_directories : list
            A list of paths to create.
        verbose : bool(optional) : defaults to True
            Controls the verbosity. If True, messages are displayed.

        Returns
        -------
        None

        Raises
        ------
        Exception

    """
    try:
        for path in path_to_directories:
            path = Path(path)
            os.makedirs(path, exist_ok=True)
            if verbose:
                logger.info(f"created directory at: {path}")
    
    except Exception as e:
        raise e