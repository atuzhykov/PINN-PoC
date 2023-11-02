from preprocessing.data_reader import *
from preprocessing.data_preparation import *
from models.GRU import *
import tensorflow as tf


def train_PINN():
    # Check if TensorFlow can access the GPU
    print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))

    # To see the device name
    if tf.test.is_gpu_available():
        print(tf.test.gpu_device_name())


    file_path = r'C:\PINN-PoC\raw_data\Oe_Zugversuch_MethodeB.xls'
    sheets_names = [f"Probe {i}" for i in range(1, 12)]
    excel_reader = ExcelReader(file_path=file_path, sheets_names=sheets_names)

    # Read the data from the Excel file
    data = excel_reader.read_data()

    # Create an instance of DataProcessor
    data_processor = DataProcessor(data)

    # Save the data to a pandas DataFrame
    df = data_processor.save_to_dataframe()

    features = ['Standardkraft (MPa)', 'Traversenweg absolut (mm)', 'Traversengeschwindigkeit (mm/min)',
                'Verfestigungsexponent ()', 'senkrechte Anisotropie ()', ' (s/Mpa)']
    target = 'Dehnung (%)'

    # Create an instance of the preprocessor class
    preprocessor = DataPreprocessor(features, target)

    # Assuming 'df' is your DataFrame
    X, y = preprocessor.fit_transform(df)

    # Split the data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    X_test = X_test.astype('float32')
    y_test = y_test.astype('float32')
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Create an instance of the GRUModel class
    gru_model = BidirectionalGRUModel(input_shape=(None, len(features)), units=100, output_size=1)

    # Train the model
    history = gru_model.train_model(X_train, y_train)

    # Evaluate the model
    performance = gru_model.evaluate_model(X_test, y_test)


if __name__ == "__main__":
    train_PINN()
