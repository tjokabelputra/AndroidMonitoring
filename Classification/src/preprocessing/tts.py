import os
import pandas as pd
import Classification.src.common.tools as tools
from sklearn.model_selection import train_test_split

def load_data(config):
    data = os.path.expanduser(
        os.path.join(config['data_dir'], "original_pca.csv")
    )

    print("Loading: ", data)
    df = pd.read_csv(data)
    print("Data shape: ", df.shape)

    return df

def split_dataset(df):
    X = df.drop(columns=['Target'])
    y = df['Target']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.4,
        random_state=42,
        stratify=y)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def save_splits(X_train, X_val, X_test, y_train, y_val, y_test, config):
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    save_dir = os.path.expanduser(config['save_dir'])
    os.mkdir(save_dir)

    train_path = os.path.join(config['save_dir'], 'train.csv')
    val_path = os.path.join(config['save_dir'], 'val.csv')
    test_path = os.path.join(config['save_dir'], 'test.csv')

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("=== SPLIT COMPLETE ===")
    print("Train:", train_df.shape)
    print("Val:  ", val_df.shape)
    print("Test: ", test_df.shape)
    print("Saved to:", save_dir)

if __name__ == "__main__":
    config = tools.load_config()

    df = load_data(config)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)
    save_splits(X_train, X_val, X_test, y_train, y_val, y_test, config)