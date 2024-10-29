import yfinance as yf
import datetime
import pandas as pd

def generate_stock_splits():
  """
  Fetches stock split data for the given symbols within the specified lookback period.

  Args:
    symbols: A list of stock symbols.
    lookback_months: A pandas Timestamp object representing the start of the lookback period.

  Returns:
    A pandas DataFrame containing the stock split data.
  """
  # Read the stock symbols from the text file
  with open("utils/symbols.txt", "r") as f:
      symbols = [line.strip() for line in f]

  # Get today's date
  today = datetime.date.today()

  # Calculate the date 60 months ago
  lookback_months = today - datetime.timedelta(days=60 * 30.5)
  lookback_months = pd.Timestamp(lookback_months, tz='America/New_York')



  all_split_data = []

  for symbol in symbols:
      ticker = yf.Ticker(symbol)
      splits = ticker.splits

      if not splits.empty:
          recent_splits = splits[splits.index >= lookback_months]

          for split_date, split_ratio in recent_splits.items():
              split_data = {
                  "Symbol": symbol,
                  "Split Date": split_date,
                  "Split Ratio": split_ratio
              }
              all_split_data.append(split_data)

  #  return pd.DataFrame(all_split_data)
  df = pd.DataFrame(all_split_data)
  df.to_csv("data/stock_splits_data.csv", index=False)

  return True


