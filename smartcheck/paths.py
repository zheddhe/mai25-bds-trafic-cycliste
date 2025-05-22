import yaml
import os
import importlib.resources

# Project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir))

def load_config():
    """
    Loads the main YAML configuration file from smartcheck.resources.

    Returns:
        dict: Contents of the YAML config.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the file content is invalid or not a dict.
    """
    try:
        config_path = importlib.resources.files("smartcheck.resources").joinpath("config.yaml")
        with config_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
            if not isinstance(data, dict):
                raise ValueError("YAML config content must be a dictionary.")
            return data
    except FileNotFoundError:
        raise FileNotFoundError("Configuration file 'config.yaml' not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML: {e}")