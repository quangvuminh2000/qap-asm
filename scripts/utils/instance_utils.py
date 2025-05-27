import numpy as np
from configparser import ConfigParser


def read_instance(file_path: str) -> np.ndarray:
    with open(file_path, "r") as file:
        return np.array([int(elem) for elem in file.read().split()])


def load_config(config_path: str) -> ConfigParser:
    """
    Load configuration from a file.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.

    Returns
    -------
    ConfigParser
        ConfigParser object containing the configuration.
    """

    cfg = ConfigParser()
    cfg.optionxform = str
    cfg.read(config_path)

    return cfg
