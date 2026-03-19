import os
import json
import numpy as np
import pandas as pd
import TimeSeries.src.common.tools as tools
from sklearn.preprocessing import StandardScaler


def load_data(config):
    data = os.path.expanduser(
        os.path.join(config["dataset_dir"], "dataset_time_series.csv")
    )

    print("Loading: ", data)
    df = pd.read_csv(data)
    print("Data Shape: ", df.shape)
    return df

def data_preparation(config, df):
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    feature_scaled = feature_scaler.fit_transform(
        df.drop(columns=['timestamp','temperature', 'battery_drain','battery_charging'])
    )
    target_scaled = target_scaler.fit_transform(
        df[['battery_drain']]
    )

    save_param(config, feature_scaler.mean_.tolist(), feature_scaler.scale_.tolist(), target_scaler.mean_.tolist(), target_scaler.scale_.tolist())
    print("Feature Scaled")
    return feature_scaled, target_scaled

def save_param(config, mean_feature, scale_feature, mean_target, scale_target):
    scaler_param = {
        "mean_feature": mean_feature,
        "scale_feature": scale_feature,
        "mean_target": mean_target,
        "scale_target": scale_target
    }

    json_dir = os.path.expanduser(
        os.path.join(config["param_dir"], "params.json")
    )
    os.makedirs(os.path.dirname(json_dir), exist_ok=True)

    with open(json_dir, "w") as f:
        json.dump(scaler_param, f)

    print("Param Saved")

def setup_data(feature, target):
    window_size = 45
    horizon = 15
    target = target.flatten()
    X = []
    y = []

    for i in range(len(feature) - window_size - horizon):
        X.append(feature[i:i+window_size])
        y.append(target[i+window_size:i+window_size+horizon])

    X = np.array(X)
    y = np.array(y)

    print("Feature Shape: ", X.shape)
    print("Label Shape: ", y.shape)
    return X, y

def train_test_split(X, y):
    train = int(len(X) * 0.6)
    val = int(len(X) * 0.8)

    X_train, y_train = X[:train], y[:train]
    X_val, y_val = X[train:val], y[train:val]
    X_test, y_test = X[val:], y[val:]

    print("Train Shape: ", X_train.shape)
    print("Validation Shape: ", X_val.shape)
    print("Test Shape: ", X_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test

def get_datasets():
    config = tools.load_config()
    df = load_data(config)
    feature, target = data_preparation(config, df)
    X, y = setup_data(feature, target)
    return train_test_split(X, y)