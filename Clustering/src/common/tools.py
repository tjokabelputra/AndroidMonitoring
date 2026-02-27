import yaml

def load_config():
    with open('Clustering/src/config.yaml') as p:
        config = yaml.safe_load(p.read())
    return config