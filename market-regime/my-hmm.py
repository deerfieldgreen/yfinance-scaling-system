import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn import hmm
import datetime
from sklearn.preprocessing import StandardScaler

# --- Data Acquisition using yfinance ---
tickers = ["SPY"]
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=700)

spy = yf.download(tickers, start=start_date, end=end_date, interval="1d")

# --- Data Preprocessing ---
data = spy.copy()

# --- Flatten the MultiIndex ---
data.columns = ['_'.join(col).strip() for col in data.columns.values] 

# --- Access columns with flattened names ---
close_prices = data["Close_SPY"]

# Calculate daily returns
daily_returns = close_prices.pct_change()

# Calculate volatility (MSE from moving average) - using bfill() directly
daily_volatility = (
    close_prices.rolling(window=10).mean().bfill() - close_prices
) ** 2

# --- Replace inf with NaN ---
daily_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
daily_volatility.replace([np.inf, -np.inf], np.nan, inplace=True)

# --- Drop missing values ---
daily_returns.dropna(inplace=True)
daily_volatility.dropna(inplace=True)

# --- Ensure both Series have the same length ---
daily_volatility = daily_volatility.loc[daily_returns.index]  # Align with daily_returns index

# --- Feature Scaling ---
scaler = StandardScaler()
scaled_data = scaler.fit_transform(np.column_stack((daily_returns, daily_volatility)))

# --- HMM Model Training ---
model = hmm.GaussianHMM(n_components=3, covariance_type="full")
model.fit(scaled_data)

# --- Regime Prediction ---
hidden_states = model.predict(scaled_data)

# --- Create a new DataFrame to store the results ---
results_df = pd.DataFrame({
    "Date": daily_returns.index,  # Use the index from daily_returns
    "Regime": hidden_states,
    "Daily_Returns": daily_returns,
    "Daily_Volatility": daily_volatility
})

# --- Print the DataFrame ---
print(results_df)




# --- Track Regime Changes ---
def track_regime_changes(df):
    regime_changes = []
    current_regime = None
    start_date = None
    for i in range(len(df)):
        regime = df["Regime"].iloc[i]
        date = df["Date"].iloc[i]
        if regime != current_regime:
            if current_regime is not None:
                regime_changes.append(
                    {
                        "Start_Date": start_date,
                        "End_Date": df["Date"].iloc[i - 1],
                        "Regime": current_regime,
                        "Duration_Days": (df["Date"].iloc[i - 1] - start_date).days,
                    }
                )
            current_regime = regime
            start_date = date
    # Add the last regime change
    if current_regime is not None:
        regime_changes.append(
            {
                "Start_Date": start_date,
                "End_Date": df["Date"].iloc[-1],
                "Regime": current_regime,
                "Duration_Days": (df["Date"].iloc[-1] - start_date).days,
            }
        )
    return pd.DataFrame(regime_changes)

regime_changes_df = track_regime_changes(results_df)

# --- Print the Regime Changes DataFrame ---
print("#######################################")
print("#######################################")
print("#######################################")
print("#######################################")
print(regime_changes_df)



