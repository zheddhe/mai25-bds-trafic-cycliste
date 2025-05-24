import os
import yaml
import importlib.resources

# Project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def get_full_path(relative_path: str) -> str:
    """
    Construct the absolute path to a file given its relative path from the project root.

    Args:
        relative_path (str): Relative path to a file from the project root.

    Returns:
        str: Absolute path to the specified file.
    """
    return os.path.join(PROJECT_ROOT, relative_path)


def load_config() -> dict:
    """
    Load the main YAML configuration file from the 'smartcheck.resources' package.

    Returns:
        dict: Parsed configuration data.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the YAML content is invalid or not a dictionary.
    """
    try:
        config_path = (
            importlib.resources.files("smartcheck.resources")
            .joinpath("config.yaml")
        )
        with config_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
            if not isinstance(data, dict):
                raise ValueError("YAML config content must be a dictionary.")
            return data
    except FileNotFoundError:
        raise FileNotFoundError("Configuration file 'config.yaml' not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML: {e}")
