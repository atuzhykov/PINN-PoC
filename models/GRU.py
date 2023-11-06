import datetime
import os

import numpy as np
from keras import Sequential, Input, Model
from keras.src.layers import Attention, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class DataPreprocessor:
    def __init__(self, dataframe, feature_names, target_name):
        self.dataframe = dataframe
        self.feature_names = feature_names
        self.target_name = target_name
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def preprocess(self):
        # Select features
        df_selected = self.dataframe[self.feature_names + [self.target_name]]

        # Normalize the dataset
        scaled_data = self.scaler.fit_transform(df_selected.values)

        # Define X and y
        X = scaled_data[:, :-1]  # All features except the target
        y = scaled_data[:, -1]  # Target feature

        # Reshape input to be [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

        # Split the data into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

    def get_train_test_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test


class TrainingPipe:
    def __init__(self, input_shape, model='GRU'):
        match model:
            case "GRU":
                self.model = self.GRUModel(input_shape)
            case "GRU_with_Attention":
                self.model = self.GRUModelWithAttention(input_shape)
            case "LSTM":
                self.model = self.LSTMModel(input_shape)
            case "LSTM_with_Attention":
                self.model = self.LSTMModelWithAttention(input_shape)

        # Compile the model
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    def GRUModel(self, input_shape):
        model = Sequential()
        # First GRU layer with Dropout
        model.add(GRU(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))  # Dropout for regularization
        model.add(BatchNormalization())  # Batch Normalization

        # Second GRU layer
        model.add(GRU(units=50, return_sequences=False))
        model.add(Dropout(0.2))  # Dropout for regularization
        model.add(BatchNormalization())  # Batch Normalization
        # Dense layer for output
        model.add(Dense(units=1))

        return model

    def GRUModelWithAttention(self, input_shape):
        # Define the input layer
        inputs = Input(shape=input_shape)

        # First GRU layer with Dropout and return sequences to feed into the attention layer
        gru_out = GRU(units=50, return_sequences=True)(inputs)
        gru_out = Dropout(0.2)(gru_out)  # Dropout for regularization
        gru_out = BatchNormalization()(gru_out)  # Batch Normalization

        # Attention layer
        # Here we are using the output of the first GRU layer as both the query and the value for the attention
        query_value = Attention()([gru_out, gru_out])

        # Second GRU layer
        gru_out_final = GRU(units=50, return_sequences=False)(query_value)
        gru_out_final = Dropout(0.2)(gru_out_final)  # Dropout for regularization
        gru_out_final = BatchNormalization()(gru_out_final)  # Batch Normalization

        # Dense layer for output
        output = Dense(units=1)(gru_out_final)

        # Compile the model
        model = Model(inputs=inputs, outputs=output)
        return model

    def LSTMModelWithAttention(self, input_shape):
        # Define the input layer
        inputs = Input(shape=input_shape)

        # First LSTM layer with Dropout and return sequences to feed into the attention layer
        lstm_out = LSTM(units=100, return_sequences=True)(inputs)
        lstm_out = Dropout(0.3)(lstm_out)  # Increased dropout for regularization
        lstm_out = BatchNormalization()(lstm_out)  # Batch Normalization

        # Attention layer
        # Here we are using the output of the first LSTM layer as both the query and the value for the attention
        query_value = Attention()([lstm_out, lstm_out])

        # Second LSTM layer with more units
        lstm_out_final = LSTM(units=100, return_sequences=False)(query_value)
        lstm_out_final = Dropout(0.3)(lstm_out_final)  # Increased dropout for regularization
        lstm_out_final = BatchNormalization()(lstm_out_final)  # Batch Normalization

        # Optional: Add additional LSTM or Dense layers if needed

        # Dense layer for output
        output = Dense(units=1)(lstm_out_final)

        # Compile the model
        model = Model(inputs=inputs, outputs=output)
        return model

    def LSTMModel(self, input_shape):
        model = Sequential()
        # First LSTM layer with Dropout and return sequences to feed into the attention layer
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))  # Dropout for regularization
        model.add(BatchNormalization())  # Batch Normalization

        # Second LSTM layer
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))  # Dropout for regularization
        model.add(BatchNormalization())  # Batch Normalization

        # Dense layer for output
        model.add(Dense(units=1))
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=4):
        # Set up the log directory for TensorBoard
        log_dir = os.path.join(
            "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.history = self.model.fit(
            X_train, y_train, epochs=epochs, batch_size=batch_size,
            validation_data=(X_val, y_val), verbose=1,
            callbacks=[tensorboard_callback]
        )

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, verbose=0)

    def predict(self, X):
        return self.model.predict(X)
