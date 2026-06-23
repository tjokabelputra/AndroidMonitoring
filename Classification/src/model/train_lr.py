"""
train_lr.py - Softmax Regression Training Script

Loads the pre-split train and validation CSV files, instantiates a Keras-based
SotmaxRegression model, trains it, and saves the resulting model to disk in
Keras format.

Model configuration (hard-coded):
    - Learning rate : 0.001
    - Max epochs    : 100
    - Batch size    : 32

This script mirrors the interface of train.py (ANN) to make it straightforward
to compare both models under identical training conditions.

Typical usage:
    python train_lr.py
"""

import os
import Classification.src.common.tools as tools
import Classification.src.data.dataio as dataio
from Classification.src.model.LR import LogisticRegression


def train(config):
    """Load data, train a logistic regression model, and save it to disk.

    Reads 'train.csv' and 'val.csv' from config['save_dir'], builds a
    SoftmaxRegression model (single Dense softmax layer), and trains it for
    up to 100 epochs. The best model weights (as determined by EarlyStopping
    inside SoftmaxRegression) are saved to config['model_dir']/LR.keras.

    Args:
        config (dict): Configuration dictionary containing:
            - 'save_dir' (str): Directory holding 'train.csv' and 'val.csv'.
            - 'model_dir' (str): Destination directory for the saved model.
              Supports '~' home-directory expansion.

    Returns:
        None
    """
    train_path = os.path.join(config['save_dir'], 'train.csv')
    val_path = os.path.join(config['save_dir'], 'val.csv')

    X_train, y_train = dataio.load(train_path)
    X_val, y_val = dataio.load(val_path)

    print("Training Softmax Regression")

    input_dim = X_train.shape[1]
    Model = LogisticRegression(
        input_dim=input_dim,
        learning_rate=0.001,
    )

    history = Model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=32,
    )

    print("Finished Training Softmax Regression")

    # Persist the trained model in Keras format
    save_path = os.path.expanduser(
        os.path.join(config['model_dir'], "LR.keras")
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Model.model.save(save_path)
    print(f"Model saved at: {save_path}")


if __name__ == "__main__":
    config = tools.load_config()
    train(config)