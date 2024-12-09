# Make sure that you have all these libaries available to run the code successf
import pandas as pd
import yfinance as yf


def target_calculate_quarter(financial_columns, num_quarters):
    counter = 1
    target_quarter_column = []
    for column in financial_columns:
        if counter <= num_quarters:
            target_quarter_column.append(column)
            counter += 1

    return target_quarter_column


def get_financial_axes_index_total_revenue(axes, axes_name):
    counter = 0
    for axe in axes:
        for axis in axe:
            if axis != axes_name:
                counter += 1
            else:
                return counter


def get_market_company_revenue(symbols_company="VWAGY"):
    # Volkswagen AG market als ticker
    company = yf.Ticker(symbols_company)

    company_short_name = company.info.get("shortName")
    print(company_short_name)

    # get Volkswagen financials per quarter

    financial_volkswagen = company.quarterly_financials

    target_quarters = target_calculate_quarter(financial_volkswagen.columns, 3)

    filter_data = []
    for target_quarter in target_quarters:
        financial_info_per_quarter = financial_volkswagen.get(target_quarter)

        company_target_infos = {
            "Period Revenue ": target_quarter,
            "Total Revenue": financial_info_per_quarter.get("Total Revenue"),
            "Company Name ": company_short_name,
        }
        filter_data.append(company_target_infos)

    print(filter_data)


list_companies = ["VWAGY", "GOOG", "AAPL", "MSFT", "TSLA", "ALV.DE"]

for company in list_companies:
    get_market_company_revenue(company)

