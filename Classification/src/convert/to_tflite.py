import os
import tensorflow as tf
import Classification.src.common.tools as tools

MODEL = "model.keras"
ANN_DIR = ""
ANN_SAVE_DIR = ""

def add_dir(config):
    global ANN_DIR, ANN_SAVE_DIR

    ANN_DIR = os.path.expanduser(
        os.path.join(config["model_dir"], MODEL)
    )

    ANN_SAVE_DIR = os.path.expanduser(
        os.path.join(config["tflite_dir"], MODEL.replace(".keras", ".tflite"))
    )

    os.makedirs(os.path.dirname(ANN_SAVE_DIR), exist_ok=True)

def convert():
    model = tf.keras.models.load_model(ANN_DIR)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(ANN_SAVE_DIR, "wb") as f:
        f.write(tflite_model)

    print("Successfully converted ANN model to TFLite")

if __name__ == "__main__":
    config = tools.load_config()
    add_dir(config)
    convert()