from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.scaler = MinMaxScaler()
        self.imputer = SimpleImputer(strategy='mean')

    def fit_transform(self, df):
        # Extract features and target variable from the DataFrame
        X = df[self.features]
        y = df[self.target]

        # Handle missing values
        X_imputed = self.imputer.fit_transform(X)

        # Normalize the features
        X_normalized = self.scaler.fit_transform(X_imputed)

        return X_normalized, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        # Split the data into training and testing sets
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def transform(self, X):
        # This method can be used to transform new data (e.g., validation, test data)
        # after the preprocessor has been fitted.
        X_imputed = self.imputer.transform(X)
        return self.scaler.transform(X_imputed)


