"""
tools.py - Common Utility Functions (Clustering)

Provides shared helper functions used across the Clustering pipeline,
such as loading the project configuration file.
"""

import yaml


def load_config():
    """Load the project configuration from the YAML file.

    Reads and parses 'Clustering/src/config.yaml' relative to the current
    working directory. The config is expected to contain keys such as
    'dataset_dir', 'param_dir', 'result_dir', and 'model_dir' used by
    various pipeline scripts.

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If 'Clustering/src/config.yaml' does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    with open('Clustering/src/config.yaml') as p:
        config = yaml.safe_load(p.read())
    return config