# Make sure that you have all these libaries available to run the code successfully
import os
import matplotlib
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import matplotlib.pyplot as plt
from requests.packages import target
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Preprocess data
def preprocess_data(data, feature='Daily Close-Open', lookback=60):
    data = data[[feature]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    x, y = [], []
    for i in range(lookback, len(scaled_data)):
        x.append(scaled_data[i - lookback:i, 0])
        y.append(scaled_data[i, 0])

    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))  # LSTM input shape
    return x, y, scaler


# Create LSTM model
def create_lstm_model(input_shape):
    keras_model = tf.keras.Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),  # 20% reduziert
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    keras_model.compile(optimizer='adam', loss='mean_squared_error')
    return keras_model

def create_rnn_model(input_shape):
    keras_model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        tf.keras.layers.SimpleRNN(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    keras_model.compile(optimizer='adam', loss='mean_squared_error')
    return keras_model
# Train and evaluate the model
def train_model(model, x_train, y_train, epochs=10, batch_size=32):
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model, history

# Predict future prices
def predict_future_price(model, data, lookback, scaler):
    scaled_data = scaler.transform(data)
    x_test = []
    for i in range(lookback, len(scaled_data)):
        x_test.append(scaled_data[i - lookback:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    train_predictions = model.predict(x_test)
    return scaler.inverse_transform(train_predictions)

# Datenaufbereitung für Analyse
def get_filter_relevant_history_open_low_close_volume(histories):
    histories['Daily Close-Open'] = histories['Close'] - histories['Open']
    return histories[['Open', 'High', 'Low','Close', 'Volume', 'Daily Close-Open']]

# period is for one year
def get_company_filter_history(symbols_company, period="1y"):
    # Autohersteller market als ticker
    company_ticker = yf.Ticker(symbols_company)
    company_short_name = company_ticker.info.get("shortName")
    print(f"Fetching data for {company_ticker.info.get('company_short_name')}...")
    history = company_ticker.history(period=period)
    filter_histories = get_filter_relevant_history_open_low_close_volume(history)
    # get Volkswagen financials per quarter
    return filter_histories, company_short_name

# Plot Funktion
def plot_results(company_name, actual_price, lstm_predictions, rnn_predictions, lookback=60):
    plt.figure(figsize=(12, 6))
    plt.plot(actual_price.index[lookback:], actual_price[target_property][lookback:],
                 label="Actual Prices")
    plt.plot(actual_price.index[lookback:], lstm_predictions, label="LSTM Predicted Prices")
    plt.plot(actual_price.index[lookback:], rnn_predictions, label="RNN Predicted Prices")
    plt.title(f'{company_name} Stock Price Prediction')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


# Plot_loss function
def plot_loss(company_name, history, model_type):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label=f'{model_type} Training Loss')
    plt.title(f'{company_name} Training Loss ({model_type})')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# List of Company
list_target_companies = ["VWAGY", "TSLA", "FORD"]
lookback = 60 # Days of historical data for prediction
target_property = 'Close'

# Loop for List of Company
for symbols_company in list_target_companies:
    get_history, company_name = get_company_filter_history(symbols_company)

    if get_history.empty:
        print(f"Data for {company_name} is empty. Skipping.")
        continue

    # Preprocess data
    x_train, y_train, scaler = preprocess_data(get_history, feature=target_property, lookback=lookback)

    # Create model
    lstm_create_model = create_lstm_model((x_train.shape[1], 1))
    rnn_create_model = create_rnn_model((x_train.shape[1], 1))

    # Train model
    lstm_model, lstm_history = train_model(lstm_create_model, x_train, y_train, epochs=10)
    rnn_model, rnn_history = train_model(rnn_create_model, x_train, y_train, epochs=10)

    # Predict future prices
    lstm_predictions = predict_future_price(lstm_model, get_history[[target_property]].values, lookback, scaler)
    rnn_predictions = predict_future_price(rnn_model, get_history[[target_property]].values, lookback, scaler)

    # Einzelne Grafiken zeigen
    plot_results(company_name, get_history, lstm_predictions, rnn_predictions, lookback)
    plot_loss(company_name, lstm_history, "LSTM")
    plot_loss(company_name, rnn_history, "RNN")
