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
        self.index_name = 'DJI'
        self.raw_series_filename = "^DJI.csv"
        self.stream_series_file = "stream_indices_data.pkl"
        self.use_recurrence_features = use_recurrence_features
        if self.use_recurrence_features:
            self.recurrence_indicators_filename = "RQA_classic_name=^DJI_window=100_step=1_rettype=6_m=1_tau=1_eps=0.csv"

    def load_data(self):
        data_processor = StreamFinanceDataProcessor(os.path.join(self.directory_path, self.stream_series_file))
        df0 = data_processor.save_to_dataframe()
        df0 = data_processor.calculate_derived_features(df0)
        df0, streaming_features = data_processor.one_index_data_preparator(df0, index_name=self.index_name,
                                                          include_derived_features=True)
        if self.use_recurrence_features:
            df1 = pd.read_csv(os.path.join(self.directory_path, self.raw_series_filename))
            df2 = pd.read_csv(os.path.join(self.directory_path, self.recurrence_indicators_filename))
            df1['Date'] = pd.to_datetime(df1['Date'])
            df2['Date'] = pd.to_datetime(df2['Date'])
            df = pd.merge(df1, df2, on='Date', how='inner')
            df = pd.merge(df0, df, on='Date', how='inner')
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            #  cyclical features
            df['day_of_week_sin'] = np.sin(df.index.dayofweek * (2. * np.pi / 7))
            df['day_of_week_cos'] = np.cos(df.index.dayofweek * (2. * np.pi / 7))

            df['month_sin'] = np.sin((df.index.month - 1) * (2. * np.pi / 12))
            df['month_cos'] = np.cos((df.index.month - 1) * (2. * np.pi / 12))

            return df, streaming_features
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


class FinanceTimeSeriesDataPreprocessor(TimeSeriesDataPreprocessor):
    def __init__(self, dataframe, feature_names, target_name, sequence_length=10):
        super().__init__(dataframe, feature_names, target_name, sequence_length)

    def preprocess(self, cutoff_date):
        # Exclude the 'Date' column from scaling
        no_scaling_data = ['day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos']
        features_to_scale = [feature for feature in self.feature_names if feature not in no_scaling_data]

        df_selected = self.dataframe[features_to_scale + [self.target_name]]
        scaled_data = self.scaler.fit_transform(df_selected)

        no_scale_data = self.dataframe[no_scaling_data]

        if no_scale_data is not None:
            scaled_data = np.concatenate([no_scale_data.values, scaled_data], axis=1)

        X, y = self.create_sequences(scaled_data)

        date_data = self.dataframe.iloc[:, 0]
        cutoff_date = pd.to_datetime(cutoff_date)
        cutoff_index = date_data.index.get_loc(cutoff_date, method='nearest')

        # Split the data at the calculated index
        self.X_train = X[:cutoff_index]
        self.X_test = X[cutoff_index:]
        self.y_train = y[:cutoff_index]
        self.y_test = y[cutoff_index:]


class StreamFinanceDataProcessor:
    def __init__(self, csv_pkl):
        self.csv_pkl = csv_pkl
        self.df = None
        self.index_names = {
            0: 'VIX',
            1: 'DJI',
            2: 'RUT',
            3: 'FTSE',
            4: 'HSI',
            5: 'GDAXI',
            6: 'IXIC',
            7: 'NYA',
            8: 'N225',
            9: 'GSPC',
        }

    def save_to_dataframe(self):

        self.df = pd.read_pickle(self.csv_pkl)

        # Convert the index to a datetime object
        self.df.index = pd.to_datetime(self.df.index)
        new_columns = []
        for col in self.df.columns:
            parts = col.split('.')
            if len(parts) < 2:
                new_col = self.index_names[0] + "_" + parts[0]
            else:
                new_col = self.index_names[int(parts[1])] + "_" + parts[0]
            new_col = new_col.replace(" ", "_")
            new_columns.append(new_col)

        self.df.columns = new_columns
        self.df.reset_index(level=0, inplace=True)
        self.df.rename(columns={'index': 'Date'}, inplace=True)
        return self.df

    def calculate_derived_features(self, df):
        def calculate_daily_returns(df, columns):
            for column in columns:
                df[column + '_Return'] = df[column].pct_change()
            return df

        # Function to calculate log returns
        def calculate_log_returns(df, columns):
            for column in columns:
                df[column + '_Log_Return'] = np.log(df[column] / df[column].shift(1))
            return df

        # Function to calculate historical volatility (rolling standard deviation of returns)
        def calculate_volatility(df, columns, window=20):
            for column in columns:
                df[column + '_Volatility'] = df[column].pct_change().rolling(window=window).std() * np.sqrt(window)
            return df

        # Function to calculate the Average True Range (ATR)
        def calculate_atr(df, high_col, low_col, close_col, window=14):
            high_low = df[high_col] - df[low_col]
            high_close = np.abs(df[high_col] - df[close_col].shift())
            low_close = np.abs(df[low_col] - df[close_col].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(window=window).mean()
            return atr

        # Calculate daily returns
        for index in self.index_names.values():
            # Calculate daily returns
            df = calculate_daily_returns(df, [f'{index}_Close'])
            # Calculate log returns
            df = calculate_log_returns(df, [f'{index}_Close'])
            # Calculate volatility
            df = calculate_volatility(df, [f'{index}_Close'])
            # Calculate ATR for each index
            df[f'{index}_ATR'] = calculate_atr(df, f'{index}_High', f'{index}_Low', f'{index}_Close')
        df.dropna(inplace=True)
        return df

    def one_index_data_preparator(self, df, index_name, include_derived_features=True):
        feature_types = ['Open', 'High', 'Low', 'Close', 'Volume']
        derived_feature_types = ['Close_Return', 'Close_Log_Return', 'Close_Volatility', 'ATR']

        basic_features = [f'{index_name}_{feature_type}' for feature_type in feature_types]
        derived_features = [f'{index_name}_{derived_feature_type}' for derived_feature_type in derived_feature_types]
        features = basic_features + derived_features if include_derived_features else basic_features
        df_selected_features = df[features + ['Date']]

        return df_selected_features, features

    def all_index_data_preparator(self, df, include_derived_features=True):
        feature_types = ['Open', 'High', 'Low', 'Close', 'Volume']
        derived_feature_types = ['Close_Return', 'Close_Log_Return', 'Close_Volatility', 'ATR']
        features = []
        for index_name in self.index_names.values():
            basic_features = [f'{index_name}_{feature_type}' for feature_type in feature_types]
            derived_features = [f'{index_name}_{derived_feature_type}' for derived_feature_type in
                                derived_feature_types]
            features += basic_features + derived_features if include_derived_features else basic_features

        df_selected_features = df[features]

        return df_selected_features, features


class PolymersTimeSeriesDataPreprocessor(TimeSeriesDataPreprocessor):
    def __init__(self, dataframe, feature_names, target_name, sequence_length=10):
        super().__init__(dataframe, feature_names, target_name, sequence_length)

    def preprocess(self):
        df_selected = self.dataframe[self.feature_names + [self.target_name]]
        scaled_data = self.scaler.fit_transform(df_selected)
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
