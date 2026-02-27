import os
import json
import pandas as pd
import Clustering.src.common.tools as tools
from sklearn.preprocessing import StandardScaler

def load_data(config):
    data = os.path.expanduser(
        os.path.join(config['dataset_dir'], "dataset.csv")
    )

    print("Loading: ", data)
    df = pd.read_csv(data)
    print("Data Shape: ", df.shape)
    return df

def check_null_and_duplicate(df):
    null = df.isnull().sum()
    duplicates = df.duplicated().sum()

    print("Null Values:\n", null)
    print("Duplicates: ", duplicates)

def remove_outliers(df):
    nums_features = df.select_dtypes(include=['number'])

    Q1 = nums_features.quantile(0.25)
    Q3 = nums_features.quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((nums_features < (Q1 - 1.5 * IQR)) | (nums_features > (Q3 + 1.5 * IQR))).any(axis=1)

    df_outlier = df[mask]
    print("Dataset After Outlier Removed: ", df_outlier.shape)
    return df_outlier

def standard_scaler(df):
    scaler = StandardScaler()

    nums_cols = df_outlier.select_dtypes(include=['int64', 'float64']).columns.to_list()
    nums_cols = [col for col in nums_cols if
                 col.lower() not in ['timestamp', 'battery_level', 'battery_charging', 'screen_on', 'brightness']]

    df[nums_cols] = scaler.fit_transform(df[nums_cols])
    print("Dataset Standard Scaled: ", df.shape)
    mean = scaler.mean_.tolist()
    scale = scaler.scale_.tolist()
    return df, mean, scale

def save_scaler_param(mean, scale):
    scaler_param = {
        "mean": mean,
        "scale": scale
    }

    json_dir = os.path.expanduser(
        os.path.join(config['param_dir'], "param.json")
    )
    os.makedirs(os.path.dirname(json_dir), exist_ok=True)

    with open(json_dir, 'w') as outfile:
        json.dump(scaler_param, outfile)

    print("Saving Scaler Param: ", json_dir)

def feature_selection(df):
    df.drop(columns=["battery_level", "brightness"], inplace=True)
    print("Feature Selection: ", df.shape)

    return df

def saved_result(df):
    save_dir = os.path.expanduser(
        os.path.join(config['dataset_dir'], "cleaned.csv")
    )
    df.to_csv(save_dir, index=False)
    print("Saved Result: ", save_dir)

if __name__ == "__main__":
    config = tools.load_config()
    df = load_data(config)
    check_null_and_duplicate(df)
    df_outlier = remove_outliers(df)
    df_scaled, mean, scale = standard_scaler(df_outlier)
    save_scaler_param(mean, scale)
    df_selection = feature_selection(df_scaled)
    saved_result(df_selection)