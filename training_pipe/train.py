import numpy as np
from models.GRU import DataPreprocessor, TrainingPipe
from preprocessing.data_reader import DataProcessor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt


def train():
    file_path = r'C:\PINN-PoC\raw_data\top_indices.csv'
    data_processor = DataProcessor(file_path)
    df = data_processor.save_to_dataframe()
    df = data_processor.calculate_derived_features(df)
    index_name = 'DJI'
    df, features = data_processor.one_index_data_preparator(df, index_name=index_name, include_derived_features=True)

    data_preprocessor = DataPreprocessor(df, feature_names=features, target_name=f'{index_name}_Close')
    data_preprocessor.preprocess()

    X_train, X_test, y_train, y_test = data_preprocessor.get_train_test_data()

    model_names = ["GRU", "GRU_with_Attention", "LSTM", "LSTM_with_Attention"]
    epochs = 300
    for model_name in model_names:
        print(f"Model {model_name} is testing on.")

        # Initialize and train the model
        model = TrainingPipe(model=model_name, input_shape=(X_train.shape[1], X_train.shape[2]))
        model.train(X_train, y_train, X_test, y_test, epochs=epochs)
        # Predict the DJI Close Price using the trained model
        predictions = model.predict(X_test)

        predicted_prices = predictions
        actual_prices = y_test
        mae = mean_absolute_error(actual_prices, predicted_prices)
        mse = mean_squared_error(actual_prices, predicted_prices)
        rmse = sqrt(mse)
        mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
        r2 = r2_score(actual_prices, predicted_prices)

        import json

        # Assuming you have the metrics calculated and stored in variables
        metrics = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "R-squared": r2
        }

        # Specify the file path where you want to save the JSON file
        file_path = f"C:\\PINN-PoC\\training_pipe\\plots\\{model_name}_metrics.json"

        # Write the dictionary to a file as JSON
        with open(file_path, 'w') as json_file:
            json.dump(metrics, json_file, indent=4)


        plt.plot(model.history.history['loss'], label='train')
        plt.plot(model.history.history['val_loss'], label='test')
        plt.title(f'Training and Validation Loss for {model_name}')
        plt.legend()
        plt.savefig(f"C:\\PINN-PoC\\training_pipe\\plots\\{model_name}_losses.png")
        plt.close()


        # Now you can plot the actual vs predicted prices
        plt.figure(figsize=(15, 7))
        plt.plot(actual_prices, label='Actual DJI Close Price', color='blue')
        plt.plot(predicted_prices, label='Predicted DJI Close Price', color='red', linestyle='--')
        plt.title(f'Comparison of Actual and Predicted DJI Close Price for {model_name}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig(f"C:\\PINN-PoC\\training_pipe\\plots\\{model_name}_actual_vs_predicted.png")
        plt.close()

if __name__ == "__main__":
    train()
