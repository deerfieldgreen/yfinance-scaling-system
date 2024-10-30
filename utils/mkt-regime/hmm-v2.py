import datetime

import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import yfinance as yf

# --- Data Acquisition using yfinance ---
tickers = ["SPY"]
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=725)

spy = yf.download(tickers, start=start_date, end=end_date, interval="1d")

# --- Data Preprocessing ---
data = spy.copy()

# Calculate daily returns
data['Daily Returns'] = data['Close'].pct_change()

# Calculate volatility using 10-day rolling MSE
data['Volatility'] = (data['Close'] - data['Close'].rolling(window=10).mean()) ** 2






# Feature scaling using StandardScaler
scaler = StandardScaler()
data[['Daily Returns', 'Volatility']] = scaler.fit_transform(
    data[['Daily Returns', 'Volatility']])

# Create DataFrame df
df = pd.DataFrame(data, index=data.index)[['Daily Returns', 'Volatility']]

# Drop rows with NaN values
df.dropna(inplace=True)

# Create and train Gaussian HMM
model = hmm.GaussianHMM(n_components=3,
                         covariance_type="full",
                         n_iter=150)  # Increased n_iter to 150
model.fit(df[['Daily Returns', 'Volatility']])

# Predict the hidden states using Viterbi algorithm
hidden_states = model.predict(df[['Daily Returns', 'Volatility']])





















# Add predicted states to DataFrame
df['Hidden States'] = hidden_states

# Group by Hidden States and calculate mean and standard deviation
regime_analysis = df.groupby('Hidden States').agg(['mean', 'std'])[['Daily Returns', 'Volatility']]

# Rename columns
regime_analysis.columns = ['Avg Daily Returns', 'Std Daily Returns', 'Avg Volatility', 'Std Volatility']

# Display results
print(regime_analysis.to_markdown(index=True, numalign="left", stralign="left"))

# --- Track Regime Changes in a DataFrame ---
regime_changes_df = pd.DataFrame(hidden_states,
                                 index=df.index,
                                 columns=['Regime'])

# Display regime changes DataFrame
print("\nRegime Changes DataFrame:")
print(regime_changes_df)



# Display regime changes DataFrame
print("\nRegime Changes DataFrame ##################### Statistics")
print( regime_changes_df.describe() )



# --- Write regime_changes_df to CSV ---
regime_changes_df.to_csv('regime_changes.csv')


