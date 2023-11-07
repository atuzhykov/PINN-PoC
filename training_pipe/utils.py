import json
import os

import numpy as np
from numpy import sqrt

from models.models import GRUModel, GRUModelWithAttention, LSTMModel, LSTMModelWithAttention
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from matplotlib import pyplot as plt


def plot_evaluations(actual_prices, flag, model, model_name, predicted_prices, sequence_length, project_dir):
    plt.plot(model.history.history['loss'], label='train')
    plt.plot(model.history.history['val_loss'], label='test')
    plt.title(f'Training and Validation Loss for {model_name} {flag} sequence length: {sequence_length}')
    plt.legend()
    save_path = os.path.join(project_dir, 'plots', f'{model_name}_{sequence_length}_{flag}_losses.png')
    plt.savefig(save_path)
    plt.close()
    # Now you can plot the actual vs predicted prices
    plt.figure(figsize=(15, 7))
    plt.plot(actual_prices, label='Actual DJI Close Price', color='blue')
    plt.plot(predicted_prices, label='Predicted DJI Close Price', color='red', linestyle='--')
    plt.title(
        f'Comparison of Actual and Predicted DJI Close Price for {model_name} {flag} sequence length: {sequence_length}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    save_path = os.path.join(project_dir, 'plots', f"{model_name}_{sequence_length}_{flag}_actual_vs_predicted.png")
    plt.savefig(save_path)
    plt.close()


def save_metrics_to_json(actual_prices, predicted_prices, model_name, flag, sequence_length, project_dir):
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
    file_path = os.path.join(project_dir, 'plots', f'{model_name}_{sequence_length}_{flag}_metrics.json')
    # Write the dictionary to a file as JSON
    with open(file_path, 'w') as json_file:
        json.dump(metrics, json_file, indent=4)


def get_model(input_shape, model_name):
    match = {"GRU": GRUModel(input_shape),
             "GRU_with_Attention": GRUModelWithAttention(input_shape),
             "LSTM": LSTMModel(input_shape),
             "LSTM_with_Attention": LSTMModelWithAttention(input_shape)}
    model = match[model_name]
    return model
