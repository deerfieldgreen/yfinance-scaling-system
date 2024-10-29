import yfinance as yf
import pandas as pd
import numpy as np

# Define the stocks to analyze
stocks = ["AAPL", "META", "TSLA", "SPY", "QQQ"]

# Create an empty DataFrame to store the results with correct data types
results = pd.DataFrame({
    "Stock": pd.Series(dtype="str"),
    "Mean Open": pd.Series(dtype=np.float64),
    "Std Dev Open": pd.Series(dtype=np.float64),
    "Mean Close": pd.Series(dtype=np.float64),
    "Std Dev Close": pd.Series(dtype=np.float64),
    "Mean Volume": pd.Series(dtype=np.float64),
    "Std Dev Volume": pd.Series(dtype=np.float64)
})

# Loop through each stock
for stock in stocks:
    # Create a Ticker object
    ticker = yf.Ticker(stock)

    # Download hourly data for the last 100 hours
    data = ticker.history(period="1d", interval="1h")  # Fetch 5 days of data to ensure 100 hours

    # Extract data for the current stock
    stock_data = data["Close"]  # Use Close for hourly data
    open_price = data["Open"]
    volume = data["Volume"]

    # Calculate mean and standard deviation
    mean_open = float(open_price.mean())
    std_open = float(open_price.std())
    mean_close = float(stock_data.mean())
    std_close = float(stock_data.std())
    mean_volume = float(volume.mean())
    std_volume = float(volume.std())

    # Append results to the DataFrame (using concat)
    new_row = pd.DataFrame({"Stock": [stock],
                              "Mean Open": [mean_open],
                              "Std Dev Open": [std_open],
                              "Mean Close": [mean_close],
                              "Std Dev Close": [std_close],
                              "Mean Volume": [mean_volume],
                              "Std Dev Volume": [std_volume]})
    results = pd.concat([results, new_row], ignore_index=True)

# Print the results
print(results.to_markdown(numalign="left", stralign="left"))