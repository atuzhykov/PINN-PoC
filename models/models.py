from keras import Sequential, Input, Model
from keras.src.layers import Attention, LSTM
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential


def GRUModel(input_shape):
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


def GRUModelWithAttention(input_shape):
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


def LSTMModelWithAttention(input_shape):
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


def LSTMModel(input_shape):
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
