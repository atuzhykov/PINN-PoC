import json
from models.models import GRUModel, GRUModelWithAttention, LSTMModel, LSTMModelWithAttention
from sklearn.metrics import mean_absolute_error, mean_squared_error

from matplotlib import pyplot as plt


def plot_evaluations(actual_prices, flag, model, model_name, predicted_prices, sequence_length):
    plt.plot(model.history.history['loss'], label='train')
    plt.plot(model.history.history['val_loss'], label='test')
    plt.title(f'Training and Validation Loss for {model_name} {flag} sequence length: {sequence_length}')
    plt.legend()
    plt.savefig(f"C:\\PINN-PoC\\training_pipe\\plots\\{model_name}_{sequence_length}_{flag}_losses.png")
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
    plt.savefig(
        f"C:\\PINN-PoC\\training_pipe\\plots\\{model_name}_{sequence_length}_{flag}_actual_vs_predicted.png")
    plt.close()


def save_metrics_to_json(actual_prices, predicted_prices, model_name, flag, sequence_length):
    mae = mean_absolute_error(actual_prices, predicted_prices)
    mse = mean_squared_error(actual_prices, predicted_prices)
    metrics = {
        "MAE": mae,
        "MSE": mse,
    }
    # Specify the file path where you want to save the JSON file
    file_path = f"C:\\PINN-PoC\\training_pipe\\plots\\{model_name}_{sequence_length}_{flag}_metrics.json"
    # Write the dictionary to a file as JSON
    with open(file_path, 'w') as json_file:
        json.dump(metrics, json_file, indent=4)


def get_model(input_shape, model_name):
    match model_name:
        case "GRU":
            model = GRUModel(input_shape)
        case "GRU_with_Attention":
            model = GRUModelWithAttention(input_shape)
        case "LSTM":
            model = LSTMModel(input_shape)
        case "LSTM_with_Attention":
            model = LSTMModelWithAttention(input_shape)
    return model
