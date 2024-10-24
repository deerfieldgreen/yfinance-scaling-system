import yfinance as yf
import datetime
import pandas as pd

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

# Get today's date
today = datetime.date.today()

# Calculate the date 50 months ago (using 30.4 days per month as an approximation)
lookback_months = today - datetime.timedelta(days=60 * 30.5)

# Convert lookback_months to a pandas Timestamp with the correct timezone
lookback_months = pd.Timestamp(lookback_months, tz='America/New_York')

# Create an empty list to store the split data
all_split_data = []

# Loop through each symbol
for symbol in symbols:
    # Download stock data
    ticker = yf.Ticker(symbol)
    splits = ticker.splits

    # Check if there are any splits at all
    if not splits.empty:
        # Filter splits to include only those within the last 50 months
        recent_splits = splits[splits.index >= lookback_months]

        # Iterate through the recent splits
        for split_date, split_ratio in recent_splits.items():
            split_data = {
                "Symbol": symbol,
                "Split Date": split_date,
                "Split Ratio": split_ratio
            }
            all_split_data.append(split_data)

# Create a DataFrame from the split data
df = pd.DataFrame(all_split_data)

# Save the DataFrame to a CSV file
df.to_csv("stock_splits_data.csv", index=False)