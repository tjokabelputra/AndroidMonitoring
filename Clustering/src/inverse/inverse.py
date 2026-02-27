import os
import json
import pandas as pd
import numpy as np
import Clustering.src.common.tools as tools

def load_scaled_dataset(config):
    data = os.path.expanduser(
        os.path.join(config['dataset_dir'], "cleaned.csv")
    )

    print("Loading: ", data)
    df = pd.read_csv(data)
    df_drop = df.drop(columns=["battery_charging", "screen_on"])
    print("Dataset Shape: ", df_drop.shape)
    return df_drop

def load_pca_label(config):
    data = os.path.expanduser(
        os.path.join(config['result_dir'], "PCA.csv")
    )

    print("Loading Label")
    df_label = pd.read_csv(data)
    labels = df_label['Target']

    return labels

def inverse_scaler(config, df):
    param_dir = os.path.expanduser(
        os.path.join(config['param_dir'], "param.json")
    )

    with open(param_dir) as f:
        params = json.load(f)

    mean  = np.array(params['mean'])
    scale = np.array(params['scale'])
    timestamp = df['timestamp'].values
    X = df.drop(columns=["timestamp"])
    X_original = X * scale + mean
    df_original = pd.DataFrame(
        X_original,
        columns=df.columns
    )
    df_original['timestamp'] = timestamp

    print("Dataset: ", df_original.columns)
    return df_original

def saved_original(df, label):
    df['Target'] = label

    save_dir = os.path.expanduser(
        os.path.join(config['result_dir'], "Original.csv")
    )

    df.to_csv(save_dir, index=False)
    print("Result Saved")

if __name__ == "__main__":
    config = tools.load_config()
    df_scaled = load_scaled_dataset(config)
    label = load_pca_label(config)
    df_original = inverse_scaler(config, df_scaled)
    saved_original(df_original, label)