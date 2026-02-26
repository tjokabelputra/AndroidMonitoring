import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class ANN:
    def __init__(self, input_dim, layer, neuron, dropout_rate, learning_rate):
        self.input_dim = input_dim
        self.layer = layer
        self.neuron = neuron
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = self.initialize(input_dim)

    def initialize(self, input_dim):
        model = models.Sequential()
        model.add(layers.Input(shape=(input_dim,)))

        for _ in range(self.layer):
            model.add(layers.Dense(self.neuron, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(self.dropout_rate))

        model.add(layers.Dense(1, activation='sigmoid'))
        model.summary()

        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            ]
        )

        return model

    def train(self, X, y, X_val, y_val, epochs, batch_size, class_weight=None):
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5
        )

        history = self.model.fit(
            X, y,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            class_weight=class_weight,
            verbose=1
        )

        return history

    def predict(self, X):
        return self.model.predict(X)