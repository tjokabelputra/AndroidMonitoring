import yaml
import pickle

def load_config():
    with open('Classification/src/config.yaml') as p:
        config = yaml.safe_load(p)
    return config