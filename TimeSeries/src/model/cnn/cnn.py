from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class CNN_RNN:
    def __init__(self, window, feature_count, horizon, f1_count, f2_count, n1_count, n2_count, learning_rate):
        self.window = window
        self.feature_count = feature_count
        self.horizon = horizon
        self.f1_count = f1_count
        self.f2_count = f2_count
        self.n1_count = n1_count
        self.n2_count = n2_count
        self.learning_rate = learning_rate
        self.model = self.init()

    def init(self):
        model = models.Sequential()
        model.add(
            layers.Conv1D(
                self.f1_count,
                kernel_size=3,
                activation="relu",
                input_shape=(self.window, self.feature_count)
            )
        )
        model.add(
            layers.Conv1D(
                self.f2_count,
                kernel_size=3,
                activation="relu",
            )
        )
        model.add(
            layers.Conv1D(
                self.f1_count,
                kernel_size=3,
                activation="relu",
            )
        )
        model.add(layers.MaxPool1D(pool_size=2))
        model.add(
            layers.SimpleRNN(
                self.n1_count,
                return_sequences=True,
                activation="tanh",
                unroll=True
            )
        )
        model.add(
            layers.SimpleRNN(
                self.n2_count,
                activation="tanh",
                unroll=True
            )
        )
        model.add(layers.Dense(self.horizon, activation="selu"))
        model.summary()

        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
        )

        return model

    def train(self, X, y, X_val, y_val, epochs, batch_size):
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5
        )

        history = self.model.fit(
            X, y,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr]
        )

        return history

    def predict(self, X_test):
        return self.model.predict(X_test)

class CNN_LSTM:
    def __init__(self, window, feature_count, horizon, f1_count, f2_count, n1_count, n2_count, learning_rate):
        self.window = window
        self.feature_count = feature_count
        self.horizon = horizon
        self.f1_count = f1_count
        self.f2_count = f2_count
        self.n1_count = n1_count
        self.n2_count = n2_count
        self.learning_rate = learning_rate
        self.model = self.init()

    def init(self):
        model = models.Sequential()
        model.add(
            layers.Conv1D(
                self.f1_count,
                kernel_size=3,
                activation="relu",
                input_shape=(self.window, self.feature_count)
            )
        )
        model.add(
            layers.Conv1D(
                self.f2_count,
                kernel_size=3,
                activation="relu",
            )
        )
        model.add(
            layers.Conv1D(
                self.f1_count,
                kernel_size=3,
                activation="relu",
            )
        )
        model.add(layers.MaxPool1D(pool_size=2))
        model.add(
            layers.LSTM(
                self.n1_count,
                return_sequences=True,
                activation="relu",
                unroll=True
            )
        )
        model.add(
            layers.LSTM(
                self.n2_count,
                activation="relu",
                unroll=True
            )
        )
        model.add(layers.Dense(self.horizon))
        model.summary()

        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
        )

        return model

    def train(self, X, y, X_val, y_val, epochs, batch_size):
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-5
        )

        history = self.model.fit(
            X, y,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr]
        )

        return history

    def predict(self, X_test):
        return self.model.predict(X_test)

class CNN_GRU:
    def __init__(self, window, feature_count, horizon, f1_count, f2_count, n1_count, n2_count, learning_rate):
        self.window = window
        self.feature_count = feature_count
        self.horizon = horizon
        self.f1_count = f1_count
        self.f2_count = f2_count
        self.n1_count = n1_count
        self.n2_count = n2_count
        self.learning_rate = learning_rate
        self.model = self.init()

    def init(self):
        model = models.Sequential()
        model.add(
            layers.Conv1D(
                self.f1_count,
                kernel_size=3,
                activation="relu",
                input_shape=(self.window, self.feature_count)
            )
        )
        model.add(
            layers.Conv1D(
                self.f2_count,
                kernel_size=3,
                activation="relu",
            )
        )
        model.add(
            layers.Conv1D(
                self.f1_count,
                kernel_size=3,
                activation="relu",
            )
        )
        model.add(layers.MaxPool1D(pool_size=2))
        model.add(
            layers.GRU(
                self.n1_count,
                return_sequences=True,
                activation="relu",
                unroll=True
            )
        )
        model.add(
            layers.GRU(
                self.n2_count,
                activation="selu",
                unroll=True
            )
        )
        model.add(layers.Dense(self.horizon))
        model.summary()

        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
        )

        return model

    def train(self, X, y, X_val, y_val, epochs, batch_size):
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-5
        )

        history = self.model.fit(
            X, y,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr]
        )

        return history

    def predict(self, X_test):
        return self.model.predict(X_test)