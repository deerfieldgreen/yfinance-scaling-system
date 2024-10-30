import datetime
import numpy as np
import pandas as pd
from hmmlearn import hmm
from scipy.stats import kstest, norm, lognorm, pareto, gamma, beta, expon
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import warnings

# --- Data Acquisition using yfinance ---
tickers = ["SPY"]
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=60)

spy = yf.download(tickers, start=start_date, end=end_date, interval="1h")

# --- Data Preprocessing ---
data = spy.copy()

# Calculate daily returns
data['Daily Returns'] = data['Close'].pct_change()

# Calculate volatility using 10-day rolling MSE
data['Volatility'] = (data['Close'] - data['Close'].rolling(window=10).mean()) ** 2

# Example: Adding momentum (e.g., past month return) and a value proxy
data['Momentum'] = data['Close'].pct_change(periods=20)  # Monthly momentum
data['Log Volume'] = np.log(data['Volume'])  # Value proxy

# Feature scaling using StandardScaler
scaler = StandardScaler()
# Standardize the additional features and include in the model training
data[['Daily Returns', 'Volatility', 'Momentum', 'Log Volume']] = scaler.fit_transform(data[['Daily Returns', 'Volatility', 'Momentum', 'Log Volume']])



# Create DataFrame df
df = pd.DataFrame(data, index=data.index)[['Daily Returns', 'Volatility']]

# Drop rows with NaN values
df.dropna(inplace=True)

# Initialize and train Gaussian HMM with explicit parameters for convergence
np.random.seed(42)  # Set a random seed to stabilize the convergence
model = hmm.GaussianHMM(
    n_components=3,
    covariance_type="full",  # full from the paper  
    n_iter=200,              # Increase iterations to help convergence
    init_params="",           # Turn off automatic parameter initialization
    params="mc"               # Fit both mean (m) and covariance (c)
)
model.startprob_ = np.array([0.5, 0.3, 0.2])
model.transmat_ = np.array([[0.8, 0.1, 0.1], 
                            [0.1, 0.8, 0.1], 
                            [0.1, 0.1, 0.8]])

model.fit(df[['Daily Returns', 'Volatility']])

# Predict the hidden states using Viterbi algorithm
hidden_states = model.predict(df[['Daily Returns', 'Volatility']])

# Add predicted states to DataFrame
df['Hidden States'] = hidden_states

# Group by Hidden States and calculate mean and standard deviation
regime_analysis = df.groupby('Hidden States').agg(['mean', 'std'])[['Daily Returns', 'Volatility']]

# Rename columns for better readability
regime_analysis.columns = ['Avg Daily Returns', 'Std Daily Returns', 'Avg Volatility', 'Std Volatility']

# Display results
print(regime_analysis.to_markdown(index=True, numalign="left", stralign="left"))






def detect_regime(observation, model):
    """
    Detects the current regime based on the given observation and HMM model.
    """
    # Define an extended set of distributions for flexibility
    distributions = [norm, gamma, beta, expon]

    # Ensure the observation is a 1D array and take only the first two values if there are extras
    observation = np.array(observation[:2]).flatten()

    # Initialize variables to track the best fit and parameters
    best_fit = None
    best_fit_pvalue = 0

    # Suppress warnings during distribution fitting
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for dist in distributions:
            try:
                # Fit the distribution to the observation
                params = dist.fit(observation)
                _, pvalue = kstest(observation, dist.cdf, args=params)
                if pvalue > best_fit_pvalue:
                    best_fit = dist
                    best_fit_pvalue = pvalue
                    best_fit_params = params
            except Exception:
                continue

    # Check if a best fit was found
    if not best_fit:
        return 2  # Default to Sideways if no valid fit found

    # Compute probability density for each state using only the first value of the observation
    regime_probabilities = []
    single_observation = observation[0]  # Use a single value for each PDF calculation
    for state in range(model.n_components):
        pdf_value = best_fit.pdf(single_observation, *best_fit_params)
        regime_probabilities.append(pdf_value)

    # Normalize probabilities
    regime_probabilities = np.array(regime_probabilities) / np.sum(regime_probabilities)

    # Extract individual probabilities for comparison
    prob_bear, prob_bull, prob_sideways = regime_probabilities

    # Detect regime with more refined probability criteria
    if prob_bear > 0.4 and (prob_bear - prob_bull) > 0.1:
        return 0  # Bear regime
    elif prob_bull > 0.4 and (prob_bull - prob_bear) > 0.1:
        return 1  # Bull regime
    else:
        return 2  # Sideways regime

# Apply the updated detect_regime function to determine regime per row
df['Detected Regime'] = df[['Daily Returns', 'Volatility']].apply(lambda x: detect_regime(x, model), axis=1)






# Print the first 5 rows of df
print(df.to_markdown(index=True, numalign="left", stralign="left"))

# Print the column names and their data types
print(df.info())





