from abc import abstractmethod

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


def calculate_mde(y_true, y_pred):
    """
    Berechnung des Mean Deviation Error (MDE):
    MDE = 1/n * Sum(|y_true - y_pred|)
    """
    n = len(y_true)
    mde = np.sum(np.abs(y_true - y_pred)) / n
    return mde


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
    return histories[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily Close-Open']]


# Custom MSE calculation
def calculate_mse(y_true, y_pred):
    """
    Berechnung des Mean Squared Error (MSE):
    MSE = 1/n * Sum((y_true - y_pred)^2)
    """
    n = len(y_true)
    mse = np.sum((y_true - y_pred) ** 2) / n
    return mse


class TrainModel:
    def __init__(self, symbols_company, target_property):
        self.company_short_name = None
        self.symbols_company = symbols_company
        self.target_property = target_property
        matplotlib.use('TkAgg')
        fig = matplotlib.pyplot.figure()
        fig.canvas.draw()

    # period is for one year
    def get_company_filter_history(self, period="2y"):
        # Autohersteller market als ticker
        company_ticker = yf.Ticker(self.symbols_company)
        self.company_short_name = company_ticker.info.get("shortName")
        print(f"Fetching data for {company_ticker.info.get('company_short_name')}...")
        history = company_ticker.history(period=period)
        filter_histories = get_filter_relevant_history_open_low_close_volume(history)
        # get Company financials per year
        return filter_histories, self.company_short_name

    # Preprocess data
    def preprocess_data(self, data, lookback=60):
        data = data[[self.target_property]].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        x, y = [], []
        for i in range(lookback, len(scaled_data)):
            x.append(scaled_data[i - lookback:i, 0])
            y.append(scaled_data[i, 0])

        x, y = np.array(x), np.array(y)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))  # LSTM input shape
        return x, y, scaler

    # Plot Funktion
    def plot_results(self, actual_price, predictions, label_predicted, lookback=60):
        plt.figure(figsize=(12, 6))
        plt.plot(actual_price.index[lookback:], actual_price[self.target_property][lookback:],
                 label="Actual Prices")
        plt.plot(actual_price.index[lookback:], predictions, label=label_predicted)
        plt.title(f'{self.company_short_name} Stock Price Prediction')
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

    # Plot_loss function
    def plot_loss(self, history, model_type):
        plt.figure(figsize=(8, 5))
        plt.plot(history.history['loss'], label=f'{model_type} Training Loss')
        plt.title(f'{self.company_short_name} Training Loss ({model_type})')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


class ITrainingModel(object):
    def __init__(self):
        pass

    @abstractmethod
    def create_model(self, input_shape):
        pass
