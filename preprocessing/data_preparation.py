import os

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler


class PolymerDataProcessor:
    def __init__(self, directory_path):
        self.directory_path = directory_path

    def load_data(self):
        # Dictionaries to hold dataframes, keyed by identifiers
        with_pi = {}
        without_pi = {}
        order = ['BDF', 'FHJ', 'JLN']
        for filename in os.listdir(self.directory_path):
            if filename.endswith(".csv"):
                for identifier in order:
                    if identifier in filename:
                        df = pd.read_csv(os.path.join(self.directory_path, filename))
                        df.rename(columns={df.columns[-1]: 'Temperature'}, inplace=True)
                        if 'PI_' in filename:
                            with_pi[identifier] = df
                        else:
                            without_pi[identifier] = df
                        break

        df_with_pi = pd.concat([with_pi[id] for id in order if id in with_pi], ignore_index=True)
        df_without_pi = pd.concat([without_pi[id] for id in order if id in without_pi], ignore_index=True)

        return df_with_pi, df_without_pi


class FinanceDataProcessor:
    def __init__(self, directory_path, use_recurrence_features=True):
        self.directory_path = directory_path
        self.raw_series_filename = "^DJI.csv"
        self.use_recurrence_features = use_recurrence_features
        if self.use_recurrence_features:
            self.recurrence_indicators_filename = "RQA_classic_name=^DJI_window=100_step=1_rettype=6_m=1_tau=1_eps=0.csv"

    def load_data(self):
        if self.use_recurrence_features:
            df1 = pd.read_csv(os.path.join(self.directory_path, self.raw_series_filename))
            df2 = pd.read_csv(os.path.join(self.directory_path, self.recurrence_indicators_filename))

            df1['Date'] = pd.to_datetime(df1['Date'])
            df2['Date'] = pd.to_datetime(df2['Date'])
            df = pd.merge(df1, df2, on='Date', how='inner')
            df['Date'] = pd.to_datetime(df['Date'])
            df['Day_Index'] = (df['Date'] - df['Date'].min()).dt.days
            return df
        else:
            df = pd.read_csv(os.path.join(self.directory_path, self.raw_series_filename))
            df['Date'] = pd.to_datetime(df['Date'])
            df['Day_Index'] = (df['Date'] - df['Date'].min()).dt.days
            return df


class TimeSeriesDataPreprocessor:
    def __init__(self, dataframe, feature_names, target_name, sequence_length=10):
        self.dataframe = dataframe
        self.feature_names = feature_names
        self.target_name = target_name
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def preprocess(self):
        # Exclude the 'Date' column from scaling
        if 'Date' in self.feature_names:
            date_data = self.dataframe['Date']
            features_to_scale = [feature for feature in self.feature_names if feature != 'Date']
        else:
            date_data = None
            features_to_scale = self.feature_names

        # Select and scale the numeric features
        df_selected = self.dataframe[features_to_scale + [self.target_name]]
        scaled_data = self.scaler.fit_transform(df_selected)

        # If 'Date' was excluded, add it back after scaling
        if date_data is not None:
            scaled_data = np.concatenate([date_data.values.reshape(-1, 1), scaled_data], axis=1)

        X, y = self.create_sequences(scaled_data)

        # Find the index to split the data based on peaks
        peaks, _ = find_peaks(y, height=None)  # You can adjust the height parameter if needed
        total_peaks = len(peaks)
        cutoff_peak = int(total_peaks * 0.8)
        cutoff_index = peaks[cutoff_peak]

        # Split the data at the calculated index
        self.X_train = X[:cutoff_index]
        self.X_test = X[cutoff_index:]
        self.y_train = y[:cutoff_index]
        self.y_test = y[cutoff_index:]

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length), :-1])
            y.append(data[i + self.sequence_length, -1])
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