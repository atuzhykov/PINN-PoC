from preprocessing.data_preparation import DataPreprocessor
from preprocessing.data_reader import DataProcessor
from training_pipe.pipeline import TrainingPipe
from training_pipe.utils import get_model, save_metrics_to_json, plot_evaluations

import os


def train():
    project_dir = os.path.abspath(os.path.dirname(__file__))
    pkl_path = os.path.join(project_dir, 'raw_data', 'top_indices.pkl')
    index_name = 'DJI'
    model_names = ["LSTM_with_Attention_v2"]
    split_date = '2018-01-01'
    sequence_length = 5
    epochs = 100
    n_step = 3
    data_processor = DataProcessor(pkl_path)
    df = data_processor.save_to_dataframe()
    df = data_processor.calculate_derived_features(df)
    use_all_data = False
    flag = 'all_indexes' if use_all_data else 'one_index'
    num_dummy_features = 90 if use_all_data else 9
    df, features = data_processor.all_index_data_preparator(df, include_derived_features=True) if use_all_data else \
        data_processor.one_index_data_preparator(df, index_name=index_name, include_derived_features=True)

    data_preprocessor = DataPreprocessor(df, feature_names=features, target_name=f'{index_name}_Close',
                                         sequence_length=sequence_length, n_steps=n_step)
    data_preprocessor.preprocess(split_date)
    X_train, X_test, y_train, y_test = data_preprocessor.get_train_test_data()
    actual_prices = data_preprocessor.inverse_transform(y_test, num_dummy_features=num_dummy_features)

    for model_name in model_names:
        print(f"Model {model_name} is testing on.")
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = get_model(input_shape, model_name, n_steps=n_step)

        model = TrainingPipe(model=model)
        model.train(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=8)

        predictions = model.predict(X_test)
        predictions = data_preprocessor.inverse_transform(predictions, num_dummy_features=num_dummy_features)
        predicted_prices = predictions

        save_metrics_to_json(actual_prices, predicted_prices, model_name, flag, n_step, sequence_length,
                             project_dir)
        plot_evaluations(actual_prices, flag, model, model_name, predicted_prices, n_step, sequence_length,
                         project_dir)


if __name__ == "__main__":
    train()
