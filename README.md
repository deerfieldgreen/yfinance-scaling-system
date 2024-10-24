# yfinance-scaling-system
yfinance-scaling-system - all data related to yfinance; we source the data and drop it in various destinations



- https://github.com/ranaroussi/yfinance/tree/main
- https://pypi.org/project/yfinance/




### Stock Splits 
- Lookback Months are/is 60

```
# Calculate the date lookback months ago (using 30.5 days per month as an approximation)
lookback_months = today - datetime.timedelta(days=60 * 30.5)

# Convert lookback_months to a pandas Timestamp with the correct timezone
lookback_months = pd.Timestamp(lookback_months, tz='America/New_York')
```

- a short list of equities: 
```

# Define the stock symbols
symbols = [
    "NVDA", 
    "MSFT", 
    "AAPL", 
    "META", 
    "AMZN", 
    "COST", 
    "TSLA", 
    "GOOGL", 
    "GOOG", 
    "BABA", 
    "PYPL",  
    "AMD", 
    "BABA",
    "AVGO"
    ]

```