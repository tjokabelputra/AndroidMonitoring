import os
import Classification.src.common.tools as tools
import Classification.src.data.dataio as dataio
from Classification.src.model.LR import LogisticRegression

def train(config):
    train_path = os.path.join(config['save_dir'], 'train.csv')
    val_path = os.path.join(config['save_dir'], 'val.csv')

    X_train, y_train = dataio.load(train_path)
    X_val, y_val = dataio.load(val_path)

    print("Training Logistic Regression")

    input_dim = X_train.shape[1]
    Model = LogisticRegression(
        input_dim = input_dim,
        learning_rate=0.001,
    )

    history = Model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=32,
    )

    print("Finished Training Logistic Regression")
    save_path = os.path.expanduser(
        os.path.join(config['model_dir'], "LR.keras")
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Model.model.save(save_path)
    print(f"Model saved at: {save_path}")

if __name__ == "__main__":
    config = tools.load_config()
    train(config)