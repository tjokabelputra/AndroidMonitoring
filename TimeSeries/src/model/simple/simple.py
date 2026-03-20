from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class RNN:
    def __init__(self, window, feature_count, horizon, n1_size, n2_size, learning_rate):
        self.window = window
        self.feature_count = feature_count
        self.horizon = horizon
        self.n1_size = n1_size
        self.n2_size = n2_size
        self.learning_rate = learning_rate
        self.model = self.initialize()

    def initialize(self):
        model = models.Sequential()
        model.add(
            layers.SimpleRNN(
                self.n1_size,
                return_sequences=True ,
                input_shape=(self.window, self.feature_count),
                activation='tanh',
                unroll=True
            )
        )
        model.add(
            layers.SimpleRNN(
                self.n2_size,
                activation='tanh',
                unroll=True
            )
        )
        model.add(layers.Dense(self.horizon))
        model.summary()

        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error',
        )

        return model

    def train(self, X, y, X_val, y_val, epochs, batch_size):
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
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
            callbacks=[early_stop, reduce_lr]
        )

        return history

    def predict(self, X_test):
        return self.model.predict(X_test)

class LSTM:
    def __init__(self, window, feature_count, horizon, n1_size, n2_size, learning_rate):
        self.window = window
        self.feature_count = feature_count
        self.horizon = horizon
        self.n1_size = n1_size
        self.n2_size = n2_size
        self.learning_rate = learning_rate
        self.model = self.initialize()

    def initialize(self):
        model = models.Sequential()
        model.add(
            layers.LSTM(
                self.n1_size,
                return_sequences=True,
                input_shape=(self.window, self.feature_count),
                activation='relu',
                unroll=True
            )
        )
        model.add(
            layers.LSTM(
                self.n2_size,
                activation='relu',
                unroll=True
            )
        )
        model.add(layers.Dense(self.horizon))
        model.summary()

        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error'
        )

        return model

    def train(self, X, y, X_val, y_val, epochs, batch_size):
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
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
            callbacks=[early_stop, reduce_lr]
        )

        return history

    def predict(self, X_test):
        return self.model.predict(X_test)

class GRU:
    def __init__(self, window, feature_count, horizon, n1_size, n2_size, learning_rate):
        self.window = window
        self.feature_count = feature_count
        self.horizon = horizon
        self.n1_size = n1_size
        self.n2_size = n2_size
        self.learning_rate = learning_rate
        self.model = self.initialize()

    def initialize(self):
        model = models.Sequential()
        model.add(
            layers.GRU(
                self.n1_size,
                return_sequences=True,
                input_shape=(self.window, self.feature_count),
                activation='relu',
                unroll=True
            )
        )
        model.add(
            layers.GRU(
                self.n2_size,
                activation='selu',
                unroll=True
            )
        )
        model.add(layers.Dense(self.horizon))
        model.summary()

        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error'
        )

        return model

    def train(self, X, y, X_val, y_val, epochs, batch_size):
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
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
            callbacks=[early_stop, reduce_lr]
        )

        return history

    def predict(self, X_test):
        return self.model.predict(X_test)