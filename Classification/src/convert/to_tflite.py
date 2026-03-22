import os
import tensorflow as tf
import Classification.src.common.tools as tools

ANN_NAME = "ANN.keras"
ANN_DIR = ""
ANN_SAVE_DIR = ""
LR_NAME = "LR.keras"
LR_DIR = ""
LR_SAVE_DIR = ""

def add_dir(config):
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

    os.makedirs(os.path.dirname(ANN_SAVE_DIR), exist_ok=True)
    os.makedirs(os.path.dirname(LR_SAVE_DIR), exist_ok=True)

def convert():
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
    add_dir(config)
    convert()