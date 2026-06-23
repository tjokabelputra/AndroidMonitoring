"""
tools.py - Common Utility Functions

Provides shared helper functions used across the Classification pipeline,
such as loading the project configuration file.
"""

import yaml
import pickle


def load_config():
    """Load the project configuration from the YAML file.

    Reads and parses 'Classification/src/config.yaml' relative to the current
    working directory. The config is expected to contain keys such as
    'data_dir', 'save_dir', 'model_dir', and 'tflite_dir' used by various
    pipeline scripts.

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If 'Classification/src/config.yaml' does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    with open('Classification/src/config.yaml') as p:
        config = yaml.safe_load(p)
    return config