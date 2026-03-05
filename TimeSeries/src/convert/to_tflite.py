import os
import tensorflow as tf
import TimeSeries.src.common.tools as tools

MODEL_DIR = ["RNN.keras", "LSTM.keras", "GRU.keras", "CNN_RNN.keras", "CNN_LSTM.keras", "CNN_GRU.keras"]
SAVE_DIR = []

def add_dir(config):
    global MODEL_DIR, SAVE_DIR

    for i in range(len(MODEL_DIR)):
        model_path = os.path.expanduser(
            os.path.join(config['model_dir'], MODEL_DIR[i])
        )

        save_path = os.path.expanduser(
            os.path.join(config['tflite_dir'], MODEL_DIR[i].replace(".keras", ".tflite"))
        )

        MODEL_DIR[i] = model_path
        SAVE_DIR.append(save_path)

def convert():
    for i in range(len(MODEL_DIR)):
        model = tf.keras.models.load_model(MODEL_DIR[i])
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()

        with open(SAVE_DIR[i], "wb") as f:
            f.write(tflite_model)

    print("Successfully Convert Model to TFLite")

if __name__ == "__main__":
    config = tools.load_config()
    add_dir(config)
    convert()