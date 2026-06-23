"""
ANN.py - Artificial Neural Network Model

Defines the ANN class, a multi-layer feedforward neural network built with
Keras for 3-class classification. Each hidden layer uses ReLU activation,
Batch Normalization, and Dropout for regularisation. The output layer uses
softmax over 3 classes.

Training uses the Adam optimiser with sparse categorical cross-entropy loss,
EarlyStopping to prevent overfitting, and ReduceLROnPlateau for adaptive
learning-rate scheduling.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class ANN:
    """Configurable multi-layer feedforward neural network for 3-class classification.

    Architecture:
        Input → [Dense(ReLU) → BatchNorm → Dropout] × layer → Dense(3, softmax)

    Attributes:
        input_dim (int): Number of input features.
        layer (int): Number of hidden layers.
        neuron (int): Number of units in each hidden layer.
        dropout_rate (float): Dropout probability applied after each hidden layer.
        learning_rate (float): Initial learning rate for the Adam optimiser.
        model (tf.keras.Model): The compiled Keras Sequential model.
    """

    def __init__(self, input_dim, layer, neuron, dropout_rate, learning_rate):
        """Initialise and compile the ANN.

        Args:
            input_dim (int): Number of input features.
            layer (int): Number of hidden Dense layers to stack.
            neuron (int): Number of units in each hidden layer.
            dropout_rate (float): Fraction of units to drop during training (0–1).
            learning_rate (float): Initial learning rate for Adam.
        """
        self.input_dim = input_dim
        self.layer = layer
        self.neuron = neuron
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = self.initialize(input_dim)

    def initialize(self, input_dim):
        """Build and compile the Keras Sequential model.

        Constructs the hidden layers based on `self.layer`, `self.neuron`, and
        `self.dropout_rate`, then appends a 3-unit softmax output layer.
        Prints a model summary after building.

        Args:
            input_dim (int): Number of input features (used to set the Input shape).

        Returns:
            tf.keras.Model: The compiled Sequential model.
        """
        model = models.Sequential()
        model.add(layers.Input(shape=(input_dim,)))

        for _ in range(self.layer):
            model.add(layers.Dense(self.neuron, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(self.dropout_rate))

        model.add(layers.Dense(3, activation='softmax'))
        model.summary()

        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            ]
        )

        return model

    def train(self, X, y, X_val, y_val, epochs, batch_size, class_weight=None):
        """Train the model with early stopping and learning-rate reduction.

        Callbacks:
            - EarlyStopping: halts training if val_loss does not improve for
              5 consecutive epochs, and restores the best weights.
            - ReduceLROnPlateau: halves the learning rate after 2 epochs of no
              val_loss improvement, with a floor of 1e-5.

        Args:
            X (array-like): Training feature matrix, shape (n_samples, input_dim).
            y (array-like): Training labels (integer-encoded), shape (n_samples,).
            X_val (array-like): Validation feature matrix.
            y_val (array-like): Validation labels (integer-encoded).
            epochs (int): Maximum number of training epochs.
            batch_size (int): Number of samples per gradient update.
            class_weight (dict, optional): Mapping of class index to weight,
                used to handle class imbalance. Defaults to None.

        Returns:
            tf.keras.callbacks.History: Keras History object containing per-epoch
                training and validation metrics.
        """
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
        """Generate class-probability predictions for the given input.

        Args:
            X (array-like): Feature matrix, shape (n_samples, input_dim).

        Returns:
            np.ndarray: Predicted probability array, shape (n_samples, 3),
                where each row sums to 1.
        """
        return self.model.predict(X)