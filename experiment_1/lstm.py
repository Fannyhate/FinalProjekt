# Make sure that you have all these libaries available to run the code successfully
import os
from abc import ABC

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout

from trainer.training import TrainModel, train_model, predict_future_price, calculate_mde, calculate_mse, \
    ITrainingModel, evaluate_baseline, baseline_prediction

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class LSTMModel(ITrainingModel, ABC):
    def __init__(self):
        super().__init__()

    # Create LSTM Model
    def create_model(self, input_shape):
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


# List of Company
list_target_companies = ["VWAGY", "TSLA", "NSANY", "HMC", "TM"]
lookback = 60  # Days of historical data for prediction
target_property = 'Close'

# Loop for List of Company
for symbols_company in list_target_companies:
    training = TrainModel(symbols_company, target_property)
    get_history, company_name = training.get_company_filter_history()

    if get_history.empty:
        print(f"Data for {company_name} is empty. Skipping.")
        continue

    # Preprocess data
    x_train, y_train, scaler = training.preprocess_data(get_history, lookback=lookback)

    # Create model
    lstm = LSTMModel()
    lstm_create_model = lstm.create_model((x_train.shape[1], 1))

    # Train model
    lstm_model, lstm_history = train_model(lstm_create_model, x_train, y_train, epochs=30)

    # Predict future prices
    lstm_predictions = predict_future_price(lstm_model, get_history[[target_property]].values, lookback,
                                            scaler)

    # Baseline Vorhersage
    baseline_preds = baseline_prediction(get_history[target_property].values, lookback)

    mde_lstm = calculate_mde(get_history[target_property][lookback:].values, lstm_predictions)
    # print(f"{company_name} MDE: {mde_lstm:.4f}")

    # Berechnung des MSE
    mse_lstm = calculate_mse(get_history[target_property][lookback:].values, lstm_predictions)

    # Baseline MSE und MDE
    baseline_mse, baseline_mde = evaluate_baseline(get_history[target_property][lookback:].values, baseline_preds)

    # Ausgabe der Ergebnisse
    print(f"{company_name} MDE (LSTM): {mde_lstm:.4f}, MSE (LSTM): {mse_lstm:.4f}")
    print(f"{company_name} MDE (Baseline): {baseline_mde:.4f}, MSE (Baseline): {baseline_mse:.4f}")

    # Einzelne Grafiken zeigen
    training.plot_results(get_history, lstm_predictions, "LSTM Predicted Prices", lookback)
    training.plot_loss(lstm_history, "LSTM")
#
