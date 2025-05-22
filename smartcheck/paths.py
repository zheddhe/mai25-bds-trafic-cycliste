import os
import yaml

# Current directory (smartcheck/)
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

# Project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))

# Useful paths
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
CONFIG_PATH = os.path.join(CONFIG_DIR, 'config.yaml')


def load_config():
    """
    Loads the main YAML configuration file.

    Returns:
        dict: Contents of the YAML file.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the YAML file is invalid.
    """
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as error:
            raise ValueError(f"YAML parsing error in {CONFIG_PATH}: {error}")