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
# Create RNN Model
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

def calculate_mse(y_true, y_pred):
    """
    Berechnung des Mean Squared Error (MSE):
    MSE = 1/n * Sum((y_true - y_pred)^2)
    """
    n = len(y_true)
    mse = np.sum((y_true - y_pred) ** 2) / n
    return mse

def calculate_mde(y_true, y_pred):
    """
    Berechnung des Mean Deviation Error (MDE):
    MDE = 1/n * Sum(|y_true - y_pred|)
    """
    n = len(y_true)
    mde = np.sum(np.abs(y_true - y_pred)) / n
    return mde

# Baseline Vorhersage: Durchschnitt der letzten 60 Schlusskurse
def baseline_prediction(data, lookback=60):
    baseline_preds = []
    for i in range(lookback, len(data)):
        # Durchschnitt der letzten 60 Schlusskurse
        avg_price = np.mean(data[i - lookback:i])
        baseline_preds.append(avg_price)
    return np.array(baseline_preds)

# Berechnung der MSE und MDE für Baseline
def evaluate_baseline(actual_price, baseline_preds):
    baseline_mse = calculate_mse(actual_price, baseline_preds)
    baseline_mde = calculate_mde(actual_price, baseline_preds)
    return baseline_mse, baseline_mde

# Datenaufbereitung für Analyse
def get_filter_relevant_history_open_low_close_volume(histories):
    histories['Daily Close-Open'] = histories['Close'] - histories['Open']
    return histories[['Open', 'High', 'Low','Close', 'Volume', 'Daily Close-Open']]

# period is for one year
def get_company_filter_history(symbols_company, period="2y"):
    # Autohersteller market als ticker
    company_ticker = yf.Ticker(symbols_company)
    company_short_name = company_ticker.info.get("shortName")
    print(f"Fetching data for {company_ticker.info.get('company_short_name')}...")
    history = company_ticker.history(period=period)
    filter_histories = get_filter_relevant_history_open_low_close_volume(history)
    # get Company per 1 year
    return filter_histories, company_short_name

# Plot Funktion
def plot_results(company_name, actual_price, rnn_predictions, lookback=60):
    plt.figure(figsize=(12, 6))
    plt.plot(actual_price.index[lookback:], actual_price[target_property][lookback:],
                 label="Actual Prices")
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
list_target_companies = ["VWAGY", "TSLA", "NSANY", "HMC", "TM"]
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

    # Create model RNN
    rnn_create_model = create_rnn_model((x_train.shape[1], 1))

    # Train model
    rnn_model, rnn_history = train_model(rnn_create_model, x_train, y_train, epochs=30)

    # Predict future prices
    rnn_predictions = predict_future_price(rnn_model, get_history[[target_property]].values, lookback, scaler)

    # Baseline Vorhersage
    baseline_preds = baseline_prediction(get_history[target_property].values, lookback)

    mde_rnn = calculate_mde(get_history[target_property][lookback:].values, rnn_predictions)
    #print(f"{company_name} MDE: {mde_rnn:.4f}")

    # Berechnung des MSE
    mse_rnn = calculate_mse(get_history[target_property][lookback:].values, rnn_predictions)
    #print(f"{company_name} MSE: {mse_rnn:.4f}")

    # Baseline MSE und MDE
    baseline_mse, baseline_mde = evaluate_baseline(get_history[target_property][lookback:].values, baseline_preds)

    # Ausgabe der Ergebnisse
    print(f"{company_name} MDE (RNN): {mde_rnn:.4f}, MSE (RNN): {mse_rnn:.4f}")
    print(f"{company_name} MDE (Baseline): {baseline_mde:.4f}, MSE (Baseline): {baseline_mse:.4f}")

    # Einzelne Grafiken zeigen
    plot_results(company_name, get_history, rnn_predictions, lookback)
    plot_loss(company_name, rnn_history, "RNN")
#