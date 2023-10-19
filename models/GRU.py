from keras.src.layers import Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import os
import time
class BidirectionalGRUModel:
    def __init__(self, input_shape, units, output_size):
        # Define model
        self.model = Sequential()
        self.model.add(Input(shape=input_shape))  # Input layer

        # Bidirectional GRU layer with 'return_sequences=True' if you plan to stack more layers
        self.model.add(Bidirectional(
            GRU(units=units, return_sequences=False, activation='tanh')))  # Set to 'True' if adding more layers
        self.model.add(Dropout(0.2))  # Adding dropout to control for overfitting
        # You can add more bidirectional or regular layers here if needed
        # Output layer
        self.model.add(Dense(output_size))
        self.model.compile(loss='mean_squared_error',
                           optimizer=Adam(learning_rate=0.001),
                           metrics=['mean_absolute_error'])
        self.model.summary()

    def train_model(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2):
        tensorboard_callback = self.create_tensorboard_callback()
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[tensorboard_callback])

    def evaluate_model(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        return self.model.predict(X)

    # Function to create a TensorBoard callback
    def create_tensorboard_callback(self):
        # Create a directory for the logs of TensorBoard
        logdir = os.path.join("logs", "fit", time.strftime("run_%Y_%m_%d-%H_%M_%S"))

        # Create a callback that logs events for TensorBoard
        return TensorBoard(
            log_dir=logdir,
            write_graph=True,  # Visualize the graph
            histogram_freq=1,  # Visualize layer statistics
            update_freq='epoch'  # Log after each epoch
        )


