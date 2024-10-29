import pandas as pd

# Load the CSV file, excluding the 'Date' column
df = pd.read_csv('forex_fx-intraday-data-15min.csv', usecols=lambda column: column != 'Date')

# Get unique symbols
symbols = df['symbol'].unique()

# Calculate and print statistics for each symbol
for symbol in symbols:
    df_symbol = df[df['symbol'] == symbol]  # Filter DataFrame for the current symbol


    print(f"\n################################################################")
    print(f"\nStatistics for Symbol: {symbol}")

    # Basic statistics
    print("\nBasic Statistics:")
    print(df_symbol.describe())

    # Additional statistics
    df_numeric = df_symbol.select_dtypes(include=['number'])
    print("\nAdditional Statistics:")
    print(df_numeric.agg(['skew', 'median']))

    # Quantiles
    print("\nQuantiles:")
    print(df_numeric.quantile([0.25, 0.5, 0.75]))


    # Other statistics
    print("\nOther Statistics:")
    for col in df_numeric.columns:
        print(f"\nColumn: {col}")
        print(f"  Variance: {df_numeric[col].var()}")
        print(f"  Kurtosis: {df_numeric[col].kurtosis()}")
        print(f"  Mode: {df_numeric[col].mode()[0]}")

