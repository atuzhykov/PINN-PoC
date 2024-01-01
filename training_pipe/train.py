import os

import pandas as pd

from preprocessing.data_preparation import PolymerDataProcessor, FinanceDataProcessor, FinanceTimeSeriesDataPreprocessor
from training_pipe.pipeline import TrainingPipe
from training_pipe.utils import get_model, plot_evaluations


def train():
    project_dir = os.path.abspath(os.path.dirname(__file__))

    # POLYMERS
    # directory_path = os.path.join(project_dir, 'raw_data', 'polymers')
    # data_processor = PolymerDataProcessor(directory_path)
    # dataset_with_pi, dataset_without_pi = data_processor.load_data()
    # datasets = {'dataset_with_pi': dataset_with_pi,
    #         'dataset_without_pi': dataset_without_pi}
    # target_name = 'Temperature'


    # FINANCE
    directory_path = os.path.join(project_dir, 'raw_data', 'finance')
    target_name = 'Adj Close'
    dataset_name = "RQA_classic_name=^DJI_window=100_step=1_rettype=6_m=1_tau=1_eps=0"
    epochs = 300
    model_names = ["LSTM_with_Attention_v1", "LSTM_with_Attention_v2", "LSTM_with_Attention_v3"]


    # From February 24 to 28, US indices showed the biggest drop since the financial crisis of 2008.
    # February 27 was the worst day for these indices since February 2018. European stock markets fell by 4-5% only on February 28.
    catastrophic_event_date = '2020-02-22'

    for use_recurrence_features in [True]:
        data_processor = FinanceDataProcessor(directory_path, use_recurrence_features=use_recurrence_features)
        dataset, streaming_features = data_processor.load_data()
        datetime_range = dataset.index.to_pydatetime()

        reccurent_feature_names = ['RecurrenceRate', 'DiagRec', 'Determinism', 'DeteRec', 'L', 'Divergence', 'LEn', 'Laminarity',
                             'TrappingTime', 'VMax', 'VEn', 'W', 'WMax', 'WEn', 'LamiDet', 'VDiv', 'WVDiv'] if use_recurrence_features else ["Day_Index"]

        sequence_lengths = [7, 13, 20, 50]
        batch_sizes = [4, 8, 16]
        lrs = [1e-2, 1e-4, 1e-5]
        n_steps = [1, 3, 5]
        if use_recurrence_features:
            for reccurent_features_num in range(0, len(reccurent_feature_names), 1):
                feature_names = reccurent_feature_names[:reccurent_features_num] + streaming_features
                print(f'feature_names {len(feature_names)}')
                print()
                for n_step in n_steps:
                    for sequence_length in sequence_lengths:
                        for batch_size in batch_sizes:
                            for lr in lrs:
                                data_preprocessor = FinanceTimeSeriesDataPreprocessor(dataset, feature_names,
                                                                                      target_name=target_name,
                                                                                      sequence_length=sequence_length)
                                data_preprocessor.preprocess(cutoff_date=catastrophic_event_date)

                                X_train, X_test, y_train, y_test = data_preprocessor.get_train_test_data()

                                for model_name in model_names:
                                    input_shape = (X_train.shape[1], X_train.shape[2])
                                    model = get_model(input_shape, model_name, n_steps=n_step)

                                    model = TrainingPipe(model=model)
                                    model.train(X_train, y_train, X_test, y_test, lr=lr, epochs=epochs, batch_size=batch_size)
                                    predictions = model.predict(X_test)

                                    plot_evaluations(y_train, y_test, predictions, model, model_name, n_step,
                                                     sequence_length, project_dir, dataset_name, batch_size, lr, reccurent_features_num, datetime_range)

        else:
            for n_step in n_steps:
                for sequence_length in sequence_lengths:
                    for batch_size in batch_sizes:
                        for lr in lrs:
                            data_preprocessor = FinanceTimeSeriesDataPreprocessor(dataset, reccurent_feature_names,
                                                                           target_name=target_name,
                                                                           sequence_length=sequence_length)
                            data_preprocessor.preprocess()

                            X_train, X_test, y_train, y_test = data_preprocessor.get_train_test_data()

                            for model_name in model_names:
                                input_shape = (X_train.shape[1], X_train.shape[2])
                                model = get_model(input_shape, model_name, n_steps=n_step)

                                model = TrainingPipe(model=model)
                                model.train(X_train, y_train, X_test, y_test, lr=lr, epochs=epochs, batch_size=batch_size)
                                predictions = model.predict(X_test)

                                plot_evaluations(y_train, y_test, predictions, model, model_name, n_step,
                                                 sequence_length, project_dir, dataset_name, batch_size, lr, 0)



if __name__ == "__main__":
    train()
