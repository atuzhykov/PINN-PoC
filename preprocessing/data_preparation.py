import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    def __init__(self, dataframe, feature_names, target_name, sequence_length=10):
        self.dataframe = dataframe
        self.feature_names = feature_names
        self.target_name = target_name
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def preprocess(self):
        # Select features
        df_selected = self.dataframe[self.feature_names + [self.target_name]]

        # Normalize the dataset
        scaled_data = self.scaler.fit_transform(df_selected.values)

        # Create sequences
        X, y = self.create_sequences(scaled_data)

        # Split the data into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length), :-1])  # All features except the target
            y.append(data[i + self.sequence_length, -1])  # Target feature
        return np.array(X), np.array(y)

    def get_train_test_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def inverse_transform(self, predictions, num_dummy_features):
        predictions = predictions.reshape(-1, 1)
        dummy_features = np.zeros((predictions.shape[0], num_dummy_features))
        predictions_with_dummy = np.concatenate([predictions, dummy_features], axis=1)
        unscaled_predictions = self.scaler.inverse_transform(predictions_with_dummy)
        unscaled_target = unscaled_predictions[:, 0]
        return unscaled_target


