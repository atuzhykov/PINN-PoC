import json
import os

import numpy as np
import pandas as pd
from numpy import sqrt

from models.models import LSTMModelWithAttention_v1, LSTMModelWithAttention_v2, LSTMModelWithAttention_v3
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from matplotlib import pyplot as plt


def plot_evaluations(train, test, predicted, model, model_name, n_step, sequence_length, project_dir, title, batch_size,
                     lr, features_num, datetime_range):
    _, _, rmse, _ = get_metrics(test, predicted)
    rmse = str(round(rmse, 4))
    plt.plot(model.history.history['loss'], label='train')
    plt.plot(model.history.history['val_loss'], label='test')
    plt.title(f'Training and Validation Loss for {model_name} sequence length: {sequence_length}')
    plt.legend()
    save_path = os.path.join(project_dir, 'plots', "finance",
                             f'{rmse}_{title}_{model_name}_sequence_length_{sequence_length}_n_steps_{n_step}_bs_{batch_size}_lr_{lr}_fn_{features_num}_losses.png')
    plt.savefig(save_path)
    plt.close()

    plt.figure(figsize=(12, 6))

    full_series = np.concatenate([train, test])
    cutoff_index = len(full_series) - len(datetime_range)
    datetime_range = datetime_range[:cutoff_index]
    plt.plot(datetime_range, full_series.flatten(), label='Full Series (Train + Test)', color='blue')

    plt.plot(datetime_range[len(train):], predicted, label='Predicted', color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    save_path = os.path.join(project_dir, 'plots', "finance",
                             f'{rmse}_{title}_{model_name}_sequence_length_{sequence_length}_n_steps_{n_step}_bs_{batch_size}_lr_{lr}_fn_{features_num}_evaluation_plot.png')
    plt.savefig(save_path)
    plt.close()


def plot_experiment(df, title):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Temperature'])
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.show()


def get_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = r2_score(actual, predicted)
    return mae, mse, rmse, mape


def get_model(input_shape, model_name, n_steps):
    match = {
        "LSTM_with_Attention_v1": LSTMModelWithAttention_v1(input_shape, n_steps=n_steps),
        "LSTM_with_Attention_v2": LSTMModelWithAttention_v2(input_shape, n_steps=n_steps),
        "LSTM_with_Attention_v3": LSTMModelWithAttention_v3(input_shape, n_steps=n_steps)
    }
    model = match[model_name]
    return model
