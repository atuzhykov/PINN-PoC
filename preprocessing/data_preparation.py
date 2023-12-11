import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class DataPreprocessor:
    def __init__(self, dataframe, feature_names, target_name, sequence_length=10, n_steps=3):
        self.dataframe = dataframe
        self.feature_names = feature_names
        self.target_name = target_name
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.n_steps = n_steps

    def preprocess(self, split_date):
        # Ensure the DataFrame is sorted by date
        self.dataframe = self.dataframe.sort_index()

        # Select features
        df_selected = self.dataframe[self.feature_names + [self.target_name]]

        # Normalize the dataset
        scaled_data = self.scaler.fit_transform(df_selected.values)

        # Create sequences
        X, y = self.create_sequences(scaled_data)

        # Find the index of the split date
        split_index = self.dataframe.index.get_loc(split_date, method='nearest')

        # Split the data using the index found
        self.X_train, self.X_test = X[:split_index], X[split_index:]
        self.y_train, self.y_test = y[:split_index], y[split_index:]

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.sequence_length - self.n_steps + 1):
            X.append(data[i:(i + self.sequence_length), :-1])  # All features except the target
            y.append(
                data[(i + self.sequence_length):(i + self.sequence_length + self.n_steps), -1])  # Next n target values
        return np.array(X), np.array(y)

    def get_train_test_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def inverse_transform(self, predictions, num_dummy_features):
        unscaled_predictions = []
        for i in range(predictions.shape[1]):  # Loop over each step
            step_predictions = predictions[:, i].reshape(-1, 1)
            dummy_features = np.zeros((step_predictions.shape[0], num_dummy_features))
            predictions_with_dummy = np.concatenate([step_predictions, dummy_features], axis=1)
            unscaled_step_predictions = self.scaler.inverse_transform(predictions_with_dummy)[:, 0]
            unscaled_predictions.append(unscaled_step_predictions)
        return np.array(unscaled_predictions).T  # Transpose to match original shape



