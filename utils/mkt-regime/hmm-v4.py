import datetime
import numpy as np
import pandas as pd
from hmmlearn import hmm
from scipy.stats import kstest, norm, gamma, beta, expon
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  # Import SimpleImputer
import yfinance as yf
import warnings

# --- Data Acquisition using yfinance ---
tickers = ["SPY"]
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=255)

spy = yf.download(tickers, start=start_date, end=end_date, interval="1h")

# --- Data Preprocessing ---
data = spy.copy()

# Calculate daily returns
data['Daily Returns'] = data['Close'].pct_change()

# Calculate volatility using 10-day rolling MSE
data['Volatility'] = (data['Close'] - data['Close'].rolling(window=10).mean()) ** 2

# Example: Adding momentum (e.g., past month return) and a value proxy
data['Momentum'] = data['Close'].pct_change(periods=20)  # Monthly momentum

# Handle zero volume before log transformation
data['Log Volume'] = np.log(data['Volume'].replace(0, 1e-10))  # Replace 0 with a small value

# Feature scaling using StandardScaler
scaler = StandardScaler()
# Standardize the additional features and include in the model training
data[['Daily Returns', 'Volatility', 'Momentum', 'Log Volume']] = scaler.fit_transform(
    data[['Daily Returns', 'Volatility', 'Momentum', 'Log Volume']])

# Create DataFrame df
df = pd.DataFrame(data, index=data.index)[['Daily Returns', 'Volatility']]

# Forward fill NaN values
df.fillna(method="ffill", inplace=True)

# --- Impute remaining NaN values using SimpleImputer ---
imputer = SimpleImputer(strategy='mean')  # Or 'median', 'most_frequent', etc.
df[['Daily Returns', 'Volatility']] = imputer.fit_transform(df[['Daily Returns', 'Volatility']])

# --- HMM Initialization with Domain Knowledge ---

# 1. Define expected ranges for each state
#    (Based on your understanding of 'Daily Returns' and 'Volatility')

# Example ranges (adjust these based on your insights)
state_ranges = {
    0: {  # Bear Market
        'Daily Returns': (-0.05, -0.01),  # Example: -5% to -1%
        'Volatility': (0.005, 0.02),  # Example: Scaled volatility range
    },
    1: {  # Bull Market
        'Daily Returns': (0.01, 0.05),  # Example: 1% to 5%
        'Volatility': (0.001, 0.01),  # Example: Scaled volatility range
    },
    2: {  # Sideways Market
        'Daily Returns': (-0.01, 0.01),  # Example: -1% to 1%
        'Volatility': (0.0, 0.005),  # Example: Scaled volatility range
    }
}

# --- Initialize the Gaussian HMM ---
np.random.seed(42)  # Set a random seed to stabilize the convergence
model = hmm.GaussianHMM(
    n_components=3,
    covariance_type="full",
    n_iter=200,
    init_params="",
    min_covar=1e-5,  # Add regularization
    params="mc"
)

# Set initial state probabilities
model.startprob_ = np.array([0.5, 0.3, 0.2])

# Set transition probabilities
model.transmat_ = np.array([[0.8, 0.1, 0.1],
                           [0.1, 0.8, 0.1],
                           [0.1, 0.1, 0.8]])

# 2. Initialize means based on the midpoints of the ranges
model.means_ = np.array([
    [np.mean(state_ranges[i]['Daily Returns']),
     np.mean(state_ranges[i]['Volatility'])]
    for i in range(model.n_components)
])

# 3. Initialize covariances (example with "full" covariance)
#    (You might need to adjust these values based on your data)
model.covars_ = np.array([
    [[np.var(state_ranges[i]['Daily Returns']) + 1e-5, 0],  # Add a small constant
     [0, np.var(state_ranges[i]['Volatility']) + 1e-5]]     # Add a small constant
    for i in range(model.n_components)
])

# --- Fit the HMM model ---
model.fit(df[['Daily Returns', 'Volatility']])
print(df[['Daily Returns', 'Volatility']].describe())

# --- Predict the hidden states using Viterbi algorithm ---
hidden_states = model.predict(df[['Daily Returns', 'Volatility']])

# --- Add predicted states to DataFrame ---
df['Hidden States'] = hidden_states

# --- Group by Hidden States and calculate mean and standard deviation ---
regime_analysis = df.groupby('Hidden States').agg(['mean', 'std'])[['Daily Returns', 'Volatility']]

# Rename columns for better readability
regime_analysis.columns = ['Avg Daily Returns', 'Std Daily Returns', 'Avg Volatility', 'Std Volatility']

# Display results
print(regime_analysis.to_markdown(index=True, numalign="left", stralign="left"))


# --- Function to detect regime ---
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

# --- Apply the updated detect_regime function to determine regime per row ---
df['Detected Regime'] = df[['Daily Returns', 'Volatility']].apply(lambda x: detect_regime(x, model), axis=1)

# --- Print the first 5 rows of df ---
print(df.head().to_markdown(index=True, numalign="left", stralign="left"))

# --- Print the column names and their data types ---
print(df.info())


# --- Save the DataFrame to a CSV file ---
df.to_csv('market_regime_data.csv')




