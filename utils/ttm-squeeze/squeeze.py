import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

yf.set_tz_cache_location(".yf-cache/")  # Cache location for yfinance

# Today's date
today = datetime.today()

# Calculate the start date as 725 days prior to today
start_date = today - timedelta(days=725)

# Format the dates as strings
start_date_str = start_date.strftime("%Y-%m-%d")
today_str = today.strftime("%Y-%m-%d")

file_path = "../../data/"
file_name ="ttm-squeeze.pkl"



# List of symbols
symbols = [
    'QQQ',
    'SPY',
    'AAPL',
    'GOOGL',
    'GOOG',
    'META',
    'MSFT',
    'AMZN',
    'TSLA'
]

# Create an empty DataFrame to store squeeze events
squeeze_df = pd.DataFrame(columns=['Symbol', 'Date', 'Close'])

# Download data for each symbol
for symbol in symbols:
    try:
        # Example of using these dates in a download function
        data = yf.download(symbol, start=start_date_str, end=today_str)
    
        if data.empty:
            print(f"No data found for {symbol}")
            continue

        # Calculate technical indicators (same as before)
        data['20sma'] = data['Close'].rolling(window=20).mean()
        data['stddev'] = data['Close'].rolling(window=20).std()
        data['lower_band'] = data['20sma'] - (2 * data['stddev'])
        data['upper_band'] = data['20sma'] + (2 * data['stddev'])

        data['TR'] = abs(data['High'] - data['Low'])
        data['ATR'] = data['TR'].rolling(window=20).mean()

        data['lower_keltner'] = data['20sma'] - (data['ATR'] * 1.5)
        data['upper_keltner'] = data['20sma'] + (data['ATR'] * 1.5)

        # Determine squeeze condition
        def in_squeeze(data):
            return (data['lower_band'] > data['lower_keltner']) & (data['upper_band'] < data['upper_keltner'])

        data['squeeze_on'] = data.apply(in_squeeze, axis=1)

        # Log squeeze events directly into the DataFrame
        for i in range(2, len(data)):
            if data['squeeze_on'].iloc[i-2] and not data['squeeze_on'].iloc[i]:
                squeeze_df = pd.concat([squeeze_df, pd.DataFrame({
                    'Symbol': [symbol],
                    'Date': [data.index[i]],
                    'Close': [data['Close'].iloc[i]]
                })], ignore_index=True)

    except Exception as e:
        print(f"Error downloading or processing data for {symbol}: {e}")



squeeze_df.to_pickle( file_path + file_name )

# Print the DataFrame
#print("\nSqueeze Events:")
#print(squeeze_df)

# Loop through the DataFrame and print contents and describe()
#for index, row in squeeze_df.iterrows():
#    print(f"\nEvent {index + 1}:")
#    print(row)

