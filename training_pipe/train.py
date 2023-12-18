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
    epochs = 200
    model_names = ["LSTM_with_Attention_v1", "LSTM_with_Attention_v2"]

    for use_recurrence_features in [False, True]:
        data_processor = FinanceDataProcessor(directory_path, use_recurrence_features=use_recurrence_features)
        dataset = data_processor.load_data()
        all_feature_names = ['RecurrenceRate', 'DiagRec', 'Determinism', 'DeteRec', 'L', 'Divergence', 'LEn', 'Laminarity',
                             'TrappingTime', 'VMax', 'VEn', 'W', 'WMax', 'WEn', 'LamiDet', 'VDiv', 'WVDiv'] if use_recurrence_features else ["Day_Index"]

        sequence_lengths = [3, 7, 13, 20]
        batch_sizes = [4, 8]
        lrs = [1e-2, 1e-5, 1e-6]
        n_steps = [1, 2]
        if use_recurrence_features:
            for features_num in range(1, len(all_feature_names), 3):
                feature_names = all_feature_names[:features_num]
            for n_step in n_steps:
                for sequence_length in sequence_lengths:
                    for batch_size in batch_sizes:
                        for lr in lrs:
                            data_preprocessor = FinanceTimeSeriesDataPreprocessor(dataset, feature_names,
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
                                                 sequence_length, project_dir, dataset_name, batch_size, lr, features_num)

        else:
            for n_step in n_steps:
                for sequence_length in sequence_lengths:
                    for batch_size in batch_sizes:
                        for lr in lrs:
                            data_preprocessor = FinanceTimeSeriesDataPreprocessor(dataset, all_feature_names,
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
                                                 sequence_length, project_dir, dataset_name, batch_size, lr, 1)



if __name__ == "__main__":
    train()
