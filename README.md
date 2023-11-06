# Stock Market Prediction using Neural Networks

## 1. Context

This project focuses on predicting stock market prices using historical data from various stock indices. We utilize advanced Deep Learning models, including Gated Recurrent Unit (GRU) networks, GRU with Attention mechanisms, and Long Short-Term Memory (LSTM) networks, to understand the temporal dependencies within stock price movements.

### Data

The dataset includes daily stock prices from multiple indices such as VIX, DJI, RUT, FTSE, HSI, GDAXI, IXIC, NYA, N225, and GSPC. Each index provides several price metrics:

- **Open**: The opening price of the stock.
- **High**: The highest price during the trading day.
- **Low**: The lowest price during the trading day.
- **Close**: The final price at which the stock trades that day.
- **Adjusted Close**: The closing price adjusted for splits and dividend distributions.
- **Volume**: The total number of shares traded during the day.

### Features and Derived Features

**Features** are the raw data points from the dataset, such as the price levels and volume.

**Derived Features** are calculated from the original features to provide additional insights to the model. These include:

- **Return**: The day-over-day percentage change in the closing price.
- **Log Return**: The natural logarithm of the return.
- **Volatility**: The statistical measure of return dispersion.
- **Average True Range (ATR)**: A volatility indicator derived from the true range.

## 2. Architecture and Training Process

### Architecture

We have developed three different neural network architectures to process the sequential data:

- **GRU**: Utilizes GRU layers to maintain memory of previous inputs.
- **GRU with Attention**: Incorporates an attention mechanism with GRU layers to focus on relevant parts of the input sequence.
- **LSTM**: Employs LSTM layers known for their ability to capture long-term dependencies.
- **LSTM with Attention**: Employs LSTM layers with attention mechanism.

Each model consists of recurrent layers followed by a dense layer to predict the stock price.

### Training Process

The training process for each model includes:

1. **Data Preprocessing**: Scaling features, creating derived features, and splitting the dataset.
2. **Model Initialization**: Based on the chosen architecture (GRU, GRU with Attention, or LSTM).
3. **Model Training**: Learning to predict stock prices using the training data.
4. **Evaluation**: Assessing performance with the test dataset and plotting the loss.
5. **Prediction**: Making predictions and inverse-scaling to compare with actual prices.
6. **Visualization**: Plotting actual vs. predicted prices for visual assessment.

### Model Selection

The `TrainingPipe` class allows for easy selection and training of the desired model architecture:

```python
training_pipe = TrainingPipe(input_shape, model='GRU_with_Attention')
training_pipe.train(X_train, y_train, X_test, y_test)
