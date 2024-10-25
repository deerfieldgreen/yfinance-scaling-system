import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression

# Download historical data
aapl = yf.download("AAPL", period="5y")
spy = yf.download("SPY", period="5y")

# Calculate daily returns
aapl['Daily_Return'] = aapl['Adj Close'].pct_change()
spy['Daily_Return'] = spy['Adj Close'].pct_change()

# Create a DataFrame with both sets of returns
df = pd.DataFrame({'AAPL_Returns': aapl['Daily_Return'], 'SPY_Returns': spy['Daily_Return']})
df = df.dropna()  # Remove rows with missing values

# Prepare data for regression
X = df['SPY_Returns'].values.reshape(-1, 1)
y = df['AAPL_Returns'].values

# Perform linear regression
model = LinearRegression()
model.fit(X, y)

# Print the results
print("Beta:", model.coef_[0])
print("Alpha:", model.intercept_)
print("R-squared:", model.score(X, y))



# Beta: 1.203753970272703
# Alpha: 0.00047515942120737254
# R-squared: 0.6277373045486645