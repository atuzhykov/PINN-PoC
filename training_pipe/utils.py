import json
import os

import numpy as np
from numpy import sqrt

from models.models import LSTMModelWithAttention_v1, LSTMModelWithAttention_v2, LSTMModelWithAttention_v3
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from matplotlib import pyplot as plt


def plot_evaluations(actual_prices, flag, model, model_name, predicted_prices, n_step, sequence_length, project_dir):

    plt.plot(model.history.history['loss'], label='train')
    plt.plot(model.history.history['val_loss'], label='test')
    plt.title(f'Training and Validation Loss for {model_name} {flag} sequence length: {sequence_length}')
    plt.legend()
    save_path = os.path.join(project_dir, 'plots', f'{model_name}_{sequence_length}_{flag}_n_steps_{n_step}_losses.png')
    plt.savefig(save_path)
    plt.close()

    plt.figure(figsize=(15, 6))

    # Plot actual prices
    plt.plot(actual_prices[:, 0].reshape(-1, 1) if n_step > 1 else actual_prices, label='Actual Prices')

    # Assuming predicted_prices is a 2D array where each column is a prediction step
    for i in range(predicted_prices.shape[1]):
        offset = sequence_length + i
        nan_array = [None] * offset
        prediction_line = nan_array + list(predicted_prices[:, i])
        prediction_line = prediction_line[:len(actual_prices)]
        plt.plot(prediction_line, label=f'Prediction Step {i+1}')

    plt.title(f'Model Evaluation for {model_name} - {flag} - n_steps {n_step}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    save_path = os.path.join(project_dir, 'plots', f'{model_name}_{sequence_length}_{flag}_n_steps_{n_step}_evaluation_plot.png')
    plt.savefig(save_path)
    plt.close()


def save_metrics_to_json(actual_prices, predicted_prices, model_name, flag, n_step, sequence_length, project_dir):
    mae = mean_absolute_error(actual_prices, predicted_prices)
    mse = mean_squared_error(actual_prices, predicted_prices)
    rmse = sqrt(mse)
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
    r2 = r2_score(actual_prices, predicted_prices)
    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R-squared": r2
    }
    # Specify the file path where you want to save the JSON file
    file_path = os.path.join(project_dir, 'plots', f'{model_name}_{sequence_length}_{flag}_n_steps_{n_step}_metrics.json')
    # Write the dictionary to a file as JSON
    with open(file_path, 'w') as json_file:
        json.dump(metrics, json_file, indent=4)


def get_model(input_shape, model_name, n_steps):
    match = {
             "LSTM_with_Attention_v1": LSTMModelWithAttention_v1(input_shape, n_steps=n_steps),
             "LSTM_with_Attention_v2": LSTMModelWithAttention_v2(input_shape, n_steps=n_steps),
             "LSTM_with_Attention_v3": LSTMModelWithAttention_v3(input_shape, n_steps=n_steps)
    }
    model = match[model_name]
    return model
