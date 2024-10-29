import yfinance as yf
import pandas as pd
from hmmlearn import hmm
import datetime

# Calculate start and end dates
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=700)

# Download SPY data from yfinance with specified dates
data = yf.download("SPY", start=start_date, end=end_date, interval="1d")

# Calculate daily returns
data['Returns'] = data['Close'].pct_change()

# Calculate volatility (MSE from moving average)
data['Volatility'] = (data['Close'] - data['Close'].rolling(window=10).mean()) ** 2

# Prepare observation data
X = data[['Returns', 'Volatility']].dropna().values

# Initialize the Gaussian HMM
model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=75)

# Train the HMM
model.fit(X)

# Get the predicted hidden states
hidden_states = model.predict(X)

# Print the model parameters
print("Initial State Probabilities:", model.startprob_)
print("State Transition Probabilities:", model.transmat_)