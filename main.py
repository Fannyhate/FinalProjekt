# Make sure that you have all these libaries available to run the code successfully
import os

import matplotlib
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


# Preprocess data
def preprocess_data(data, feature='Daily Close-Open', lookback=60):
    data = data[[feature]].values
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = min_max_scaler.fit_transform(data)

    x, y = [], []
    for i in range(lookback, len(scaled_data)):
        x.append(scaled_data[i - lookback:i, 0])
        y.append(scaled_data[i, 0])

    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))  # LSTM input shape
    return x, y, min_max_scaler


# Create LSTM model
def create_lstm_model(input_shape):
    keras_model = tf.keras.Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    keras_model.compile(optimizer='adam', loss='mean_squared_error')
    return keras_model


# Train and evaluate the model
def train_model(model, x_train, y_train, epochs=50, batch_size=32):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model


# Predict future prices
def predict_future_price(model, data, lookback, scaler):
    scaled_data = scaler.transform(data)
    x_test = []
    for i in range(lookback, len(scaled_data)):
        x_test.append(scaled_data[i - lookback:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    train_predictions = model.predict(x_test)
    train_predictions = scaler.inverse_transform(train_predictions)
    return train_predictions


def get_filter_relevant_history_open_low_close_volume(histories):
    dict_histories = histories[['Open', 'High', 'Low', 'Close', 'Volume']].to_dict('records')
    list_of_dates = []
    for date in histories.axes:
        for value in date.values:
            if type(value) is not str:
                time_str = np.datetime_as_string(value, unit='D')
                list_of_dates.append(time_str)

    columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date', 'Daily Close-Open']
    counter = 0

    for dict_history in dict_histories:
        dict_history['Date'] = list_of_dates[counter]
        diff = dict_history['Close'] - dict_history['Open']
        dict_history['Daily Close-Open'] = diff
        counter += 1

    filter_dp = pd.DataFrame(dict_histories, columns=columns)
    return filter_dp

# period is for one year
def get_company_filter_history(symbols_company="VWAGY", period="1y"):
    # Volkswagen AG market als ticker
    company_ticker = yf.Ticker(symbols_company)

    company_short_name = company_ticker.info.get("shortName")
    print(company_short_name)
    histories_company = company_ticker.history(period=period)
    filter_histories = get_filter_relevant_history_open_low_close_volume(histories_company)
    print(filter_histories)
    # get Volkswagen financials per quarter

    return filter_histories, company_short_name


list_target_companies = ["VWAGY"]

target_property = 'Close'
actual_property_label = 'Actual Prices'
predicted_property_label = 'Predicted Prices'

matplotlib.use('TkAgg')
fig = matplotlib.pyplot.figure()
fig.canvas.draw()

for company in list_target_companies:
    get_history, company_name = get_company_filter_history(company)
    lookback = 60  # Days of historical data for prediction

    # Preprocess data
    x_train, y_train, scaler = preprocess_data(get_history, feature=target_property, lookback=lookback)

    # Create model
    create_model = create_lstm_model((x_train.shape[1], 1))

    # Train model
    model = train_model(create_model, x_train, y_train, epochs=10, batch_size=32)

    # Predict future prices
    predictions = predict_future_price(model, get_history[[target_property]].values, lookback, scaler)

    # Plot results

    plt.figure(figsize=(12, 6))
    plt.plot(get_history.index[lookback:], get_history[target_property].iloc[lookback:], label=actual_property_label)
    plt.plot(get_history.index[lookback:], predictions, label=predicted_property_label)
    plt.title(f"{company_name} Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
