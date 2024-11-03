import yfinance as yf
import pandas as pd

yf.set_tz_cache_location(".yf.cache/")  # Cache location for yfinance

# Set the start and end dates
start_date = '2024-01-01'
end_date = '2024-10-29'

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
        data = yf.download(symbol, start=start_date, end=end_date)

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

# Print the DataFrame
print("\nSqueeze Events:")
print(squeeze_df)

# Loop through the DataFrame and print contents and describe()
for index, row in squeeze_df.iterrows():
    print(f"\nEvent {index + 1}:")
    print(row)

