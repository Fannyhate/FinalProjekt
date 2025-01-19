# Make sure that you have all these libaries available to run the code successfully
import os
from abc import ABC

import tensorflow as tf
from keras.src.layers import SimpleRNN
from tensorflow.keras.layers import Dense, Dropout

from trainer.training import TrainModel, calculate_mde, predict_future_price, train_model, calculate_mse, ITrainingModel, evaluate_baseline, baseline_prediction

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


# class inheritance from ItrainingModel
class RNNModel(ITrainingModel, ABC):
    def __init__(self):
        super().__init__()

    # Create RNN Model
    def create_model(self, input_shape):
        keras_model = tf.keras.Sequential([
            SimpleRNN(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            SimpleRNN(50, return_sequences=False),
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

    # Create model RNN
    rnn = RNNModel()

    rnn_create_model = rnn.create_model((x_train.shape[1], 1))

    # Train model
    rnn_model, rnn_history = train_model(rnn_create_model, x_train, y_train, epochs=30)

    # Predict future prices
    rnn_predictions = predict_future_price(rnn_model, get_history[[target_property]].values, lookback, scaler)

    # Baseline Vorhersage
    baseline_preds = baseline_prediction(get_history[target_property].values, lookback)

    mde_rnn = calculate_mde(get_history[target_property][lookback:].values, rnn_predictions)

    mde = calculate_mde(get_history[target_property][lookback:].values, rnn_predictions)
    print(f"{company_name} MDE: {mde:.4f}")

    # Berechnung des MSE
    mse = calculate_mse(get_history[target_property][lookback:].values, rnn_predictions)
    print(f"{company_name} MSE: {mse:.4f}")

    mse_rnn = calculate_mse(get_history[target_property][lookback:].values, rnn_predictions)

    # Baseline MSE und MDE
    baseline_mse, baseline_mde = evaluate_baseline(get_history[target_property][lookback:].values, baseline_preds)

    # Ausgabe der Ergebnisse
    print(f"{company_name} MDE (RNN): {mde_rnn:.4f}, MSE (RNN): {mse_rnn:.4f}")
    print(f"{company_name} MDE (Baseline): {baseline_mde:.4f}, MSE (Baseline): {baseline_mse:.4f}")

    # Einzelne Grafiken zeigen
    training.plot_results(get_history, rnn_predictions, "RNN Predicted Prices", lookback)
    training.plot_loss(rnn_history, "RNN")
#
