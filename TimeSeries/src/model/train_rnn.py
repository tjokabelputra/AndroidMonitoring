import os
import TimeSeries.src.common.tools as tools
from TimeSeries.src.model.simple.simple import RNN
from TimeSeries.src.preprocessing.preprocessing import get_datasets

def train(config):
    X_train, y_train, X_val, y_val, X_test, y_test = get_datasets()

    print("Training RNN")
    window = X_train.shape[1]
    feature_count = X_train.shape[2]
    horizon = y_train.shape[1]
    Model = RNN(
        window=window,
        feature_count=feature_count,
        horizon=horizon,
        n1_size=20,
        n2_size=10,
        learning_rate=0.001,
    )

    history = Model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=32
    )

    print("Finished Training RNN")
    save_path = os.path.expanduser(
        os.path.join(config["model_dir"], "RNN.keras")
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Model.model.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    config = tools.load_config()
    train(config)