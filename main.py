# Make sure that you have all these libaries available to run the code successfully
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import keras


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


def get_company_filter_history(symbols_company="VWAGY", period="1mo"):
    # Volkswagen AG market als ticker
    company_ticker = yf.Ticker(symbols_company)

    company_short_name = company_ticker.info.get("shortName")
    print(company_short_name)
    histories_company = company_ticker.history(period=period)
    filter_histories = get_filter_relevant_history_open_low_close_volume(histories_company)
    print(filter_histories)
    # get Volkswagen financials per quarter

    return filter_histories


list_target_companies = ["VWAGY"]

for company in list_target_companies:
    get_history = get_company_filter_history(company)
