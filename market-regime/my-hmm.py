import yfinance as yf
import pandas as pd
from hmmlearn import hmm

# Download SPY data from Yahoo Finance
spy = yf.download("SPY", period="725d")

# Calculate log returns
spy["log_returns"] = np.log(spy["Adj Close"] / spy["Adj Close"].shift(1))
spy.dropna(inplace=True)

# Fit a 2-state Hidden Markov Model
model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)
model.fit(spy[["log_returns"]])

# Predict the hidden states (market regimes)
spy["state"] = model.predict(spy[["log_returns"]])

# Print the model parameters
print("Transition matrix:")
print(model.transmat_)
print("\nMeans and stds of each state:")
for i in range(model.n_components):
    print(f"State {i}: Mean = {model.means_[i][0]:.4f}, Std = {np.sqrt(model.covars_[i][0][0]):.4f}")

# Plot the results
spy["state"].plot(figsize=(12, 6), title="Market Regimes")