import numpy as np
import pandas as pd


class DataProcessor:
    def __init__(self, pkl_path):
        self.pkl_path = pkl_path
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

        self.df = pd.read_pickle(self.pkl_path)

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

    def one_index_data_preparator(self, df, index_name, include_derived_features=True ):
        feature_types = ['Open', 'High', 'Low', 'Close', 'Volume']
        derived_feature_types = ['Close_Return', 'Close_Log_Return', 'Close_Volatility', 'ATR']

        basic_features = [f'{index_name}_{feature_type}' for feature_type in feature_types]
        derived_features = [f'{index_name}_{derived_feature_type}' for derived_feature_type in derived_feature_types]
        features = basic_features + derived_features if include_derived_features else basic_features

        df_selected_features = df[features]

        return df_selected_features, features

    def all_index_data_preparator(self, df, include_derived_features=True ):
        feature_types = ['Open', 'High', 'Low', 'Close', 'Volume']
        derived_feature_types = ['Close_Return', 'Close_Log_Return', 'Close_Volatility', 'ATR']
        features = []
        for index_name in self.index_names.values():
            basic_features = [f'{index_name}_{feature_type}' for feature_type in feature_types]
            derived_features = [f'{index_name}_{derived_feature_type}' for derived_feature_type in derived_feature_types]
            features += basic_features + derived_features if include_derived_features else basic_features

        df_selected_features = df[features]

        return df_selected_features, features





