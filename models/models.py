from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Attention, LSTM
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.python.keras.layers import Bidirectional

def LSTMModelWithAttention_v2(input_shape, n_steps):
    # Define the input layer
    inputs = Input(shape=input_shape)

    # First Bidirectional LSTM layer with Dropout and return sequences
    lstm_out = Bidirectional(LSTM(units=128, return_sequences=True))(inputs)
    lstm_out = Dropout(0.4)(lstm_out)
    lstm_out = BatchNormalization()(lstm_out)

    # Second Bidirectional LSTM layer
    lstm_out = Bidirectional(LSTM(units=128, return_sequences=True))(lstm_out)
    lstm_out = Dropout(0.4)(lstm_out)
    lstm_out = BatchNormalization()(lstm_out)

    # Attention layer
    query_value = Attention()([lstm_out, lstm_out])

    # Third LSTM layer
    lstm_out_final = LSTM(units=128, return_sequences=False)(query_value)
    lstm_out_final = Dropout(0.4)(lstm_out_final)
    lstm_out_final = BatchNormalization()(lstm_out_final)

    # Optional: Additional Dense layers
    dense_out = Dense(units=64, activation='relu')(lstm_out_final)
    dense_out = Dropout(0.4)(dense_out)

    # Dense layer for output
    output = Dense(units=n_steps)(dense_out)

    # Compile the model
    model = Model(inputs=inputs, outputs=output)
    return model


def LSTMModelWithAttention_v1(input_shape, n_steps):
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
    output = Dense(units=n_steps)(lstm_out_final)

    # Compile the model
    model = Model(inputs=inputs, outputs=output)
    return model


def LSTMModelWithAttention_v3(input_shape, n_steps):
    inputs = Input(shape=input_shape)

    lstm_out = Bidirectional(LSTM(units=256, return_sequences=True))(inputs)
    lstm_out = Dropout(0.4)(lstm_out)
    lstm_out = BatchNormalization()(lstm_out)

    lstm_out = Bidirectional(LSTM(units=256, return_sequences=True))(lstm_out)
    lstm_out = Dropout(0.4)(lstm_out)
    lstm_out = BatchNormalization()(lstm_out)

    query_value = Attention()([lstm_out, lstm_out])

    lstm_out_final = LSTM(units=256, return_sequences=False)(query_value)
    lstm_out_final = Dropout(0.4)(lstm_out_final)
    lstm_out_final = BatchNormalization()(lstm_out_final)

    dense_out = Dense(units=128, activation='relu')(lstm_out_final)
    dense_out = Dropout(0.4)(dense_out)
    dense_out = Dense(units=64, activation='relu')(dense_out)
    dense_out = Dropout(0.4)(dense_out)

    output = Dense(units=n_steps)(dense_out)

    model = Model(inputs=inputs, outputs=output)
    return model
