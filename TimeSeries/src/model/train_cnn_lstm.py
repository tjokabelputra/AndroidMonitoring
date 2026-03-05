import os
import TimeSeries.src.common.tools as tools
from TimeSeries.src.model.cnn.cnn import CNN_LSTM
from TimeSeries.src.preprocessing.preprocessing import get_datasets

def train(config):
    X_train, y_train, X_val, y_val, X_test, y_test = get_datasets()

    print("Training CNN LSTM")
    window = X_train.shape[1]
    feature_count = X_train.shape[2]
    horizon = y_train.shape[1]
    Model = CNN_LSTM(
        window = window,
        feature_count = feature_count,
        horizon = horizon,
        f1_count=4,
        f2_count=8,
        n1_count=5,
        n2_count=5,
        learning_rate=0.001,
    )

    history = Model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=32
    )

    print("Finished Training CNN LSTM")
    save_dir = os.path.expanduser(
        os.path.join(config['model_dir'], 'CNN_LSTM.keras')
    )
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    Model.model.save(save_dir)
    print(f"Model saved to {save_dir}")

if __name__ == "__main__":
    config = tools.load_config()
    train(config)