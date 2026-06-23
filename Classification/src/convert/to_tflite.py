"""
to_tflite.py - Keras to TFLite Conversion Script

Converts trained ANN and Logistic Regression Keras models (.keras) to
TensorFlow Lite format (.tflite) for deployment on resource-constrained
devices. Output paths are derived from the project config and created
automatically if they do not exist.

Module-level path variables (ANN_DIR, ANN_SAVE_DIR, LR_DIR, LR_SAVE_DIR) are
populated at runtime by calling add_dir(config) before convert().

Typical usage:
    python to_tflite.py
"""

import os
import tensorflow as tf
import Classification.src.common.tools as tools

# Filenames of the source Keras models (populated by add_dir)
ANN_NAME = "ANN.keras"
ANN_DIR = ""       # Resolved path to the ANN .keras model
ANN_SAVE_DIR = ""  # Resolved path for the output ANN .tflite file

LR_NAME = "LR.keras"
LR_DIR = ""        # Resolved path to the LR .keras model
LR_SAVE_DIR = ""   # Resolved path for the output LR .tflite file


def add_dir(config):
    """Resolve and set all model input/output paths from the config.

    Populates the module-level path variables (ANN_DIR, ANN_SAVE_DIR, LR_DIR,
    LR_SAVE_DIR) using the directories specified in `config`. Output
    directories are created if they do not already exist.

    Args:
        config (dict): Configuration dictionary containing:
            - 'model_dir' (str): Directory holding the trained .keras models.
            - 'tflite_dir' (str): Destination directory for the .tflite files.
              Both support '~' home-directory expansion.

    Returns:
        None
    """
    global ANN_DIR, ANN_SAVE_DIR, LR_DIR, LR_SAVE_DIR

    ANN_DIR = os.path.expanduser(
        os.path.join(config["model_dir"], ANN_NAME)
    )

    ANN_SAVE_DIR = os.path.expanduser(
        os.path.join(config["tflite_dir"], ANN_NAME.replace(".keras", ".tflite"))
    )

    LR_DIR = os.path.expanduser(
        os.path.join(config["model_dir"], LR_NAME)
    )

    LR_SAVE_DIR = os.path.expanduser(
        os.path.join(config["tflite_dir"], LR_NAME.replace(".keras", ".tflite"))
    )

    # Ensure output directories exist before conversion
    os.makedirs(os.path.dirname(ANN_SAVE_DIR), exist_ok=True)
    os.makedirs(os.path.dirname(LR_SAVE_DIR), exist_ok=True)


def convert():
    """Load both Keras models and convert them to TFLite format.

    Reads the ANN and LogisticRegression models from the paths set by
    add_dir(), converts each using TFLiteConverter, and writes the resulting
    binary .tflite files to their respective save paths.

    Must be called after add_dir() has been executed, as it relies on the
    module-level path variables being populated.

    Returns:
        None

    Raises:
        OSError: If the source .keras files cannot be found at ANN_DIR or LR_DIR.
    """
    model_ann = tf.keras.models.load_model(ANN_DIR)
    model_lr = tf.keras.models.load_model(LR_DIR)

    converter_ann = tf.lite.TFLiteConverter.from_keras_model(model_ann)
    converter_lr = tf.lite.TFLiteConverter.from_keras_model(model_lr)

    tflite_model_ann = converter_ann.convert()
    tflite_model_lr = converter_lr.convert()

    with open(ANN_SAVE_DIR, "wb") as f:
        f.write(tflite_model_ann)

    print("Successfully converted ANN model to TFLite")

    with open(LR_SAVE_DIR, "wb") as f:
        f.write(tflite_model_lr)

    print("Successfully converted Logistic Regression model to TFLite")


if __name__ == "__main__":
    config = tools.load_config()
    add_dir(config)   # Resolve paths from config before converting
    convert()