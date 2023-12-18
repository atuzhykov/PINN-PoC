import datetime
import os

from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import LearningRateScheduler


class TrainingPipe:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train, X_val, y_val, lr=1e-4, epochs=100, batch_size=4):
        self.model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
        def scheduler(epoch, lr):
            if epoch < 10:
                return lr
            else:
                return lr * tf.math.exp(-0.1)

        lr_scheduler_callback = LearningRateScheduler(scheduler)
        log_dir = os.path.join(
            "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.history = self.model.fit(
            X_train, y_train, epochs=epochs, batch_size=batch_size,
            validation_data=(X_val, y_val), verbose=1,
            callbacks=[tensorboard_callback, lr_scheduler_callback]
        )

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, verbose=0)

    def predict(self, X):
        return self.model.predict(X)
