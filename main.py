import yfinance as yf
# Microsoft market
dat = yf.Ticker("MSFT")

print(dat.info)
financial = dat.financials
print(financial.info)

# get historical market data
market_history = dat.history(period='6mo')
print(market_history)


print("----------- get history market of Tesla-------------------------------------------------------------")
# Tesla market
dat = yf.Ticker("TSLA")

print(dat.info)
financial = dat.financials
print(financial.info)

# get historical market data
market_history = dat.history(period='6mo')
print(market_history)

