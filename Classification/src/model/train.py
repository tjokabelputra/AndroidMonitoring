"""
train.py - ANN Training Script

Loads the pre-split train and validation CSV files, instantiates an ANN model
with a fixed architecture, trains it, and saves the resulting model to disk in
Keras format.

Model configuration (hard-coded):
    - Hidden layers : 1
    - Units per layer: 32
    - Dropout rate  : 0.0 (disabled)
    - Learning rate : 0.001
    - Max epochs    : 100
    - Batch size    : 32

Typical usage:
    python train.py
"""

import os
import Classification.src.common.tools as tools
import Classification.src.data.dataio as dataio
from Classification.src.model.ANN import ANN


def train(config):
    """Load data, train an ANN, and save the model to disk.

    Reads 'train.csv' and 'val.csv' from config['save_dir'], builds an ANN
    with a fixed single hidden layer of 32 units, and trains it for up to 100
    epochs. The best model weights (as determined by EarlyStopping inside ANN)
    are saved to config['model_dir']/ANN.keras.

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

    print("Training ANN")

    input_dim = X_train.shape[1]
    Model = ANN(
        input_dim=input_dim,
        layer=1,
        neuron=32,
        dropout_rate=0.0,
        learning_rate=0.001,
    )

    history = Model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=32,
    )

    print("Finished Training ANN")

    # Persist the trained model in Keras format
    save_path = os.path.expanduser(
        os.path.join(config['model_dir'], "ANN.keras")
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Model.model.save(save_path)
    print(f"Model saved at: {save_path}")


if __name__ == "__main__":
    config = tools.load_config()
    train(config)