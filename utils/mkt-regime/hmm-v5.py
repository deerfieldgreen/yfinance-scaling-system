import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn import hmm
from scipy.stats import kstest, norm, gamma, beta, expon
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import yfinance as yf
import warnings

# --- Data Acquisition using yfinance ---
tickers = ["SPY"]
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=(350 * 1))  # Increased dataset size

spy = yf.download(tickers, start=start_date, end=end_date, interval="1h")

# --- Data Preprocessing ---
data = spy.copy()

# Calculate daily returns
data['Daily Returns'] = data['Close'].pct_change()

# Calculate volatility using 10-day rolling MSE
data['Volatility'] = (data['Close'] - data['Close'].rolling(window=10).mean()) ** 2

# Example: Adding momentum (e.g., past month return)
data['Momentum'] = data['Close'].pct_change(periods=20)

# Handle zero volume before log transformation
data['Log Volume'] = np.log(data['Volume'].replace(0, 1e-10))

# Feature scaling using StandardScaler
scaler = StandardScaler()
data[['Daily Returns', 'Volatility', 'Momentum', 'Log Volume']] = scaler.fit_transform(
    data[['Daily Returns', 'Volatility', 'Momentum', 'Log Volume']])

# Create DataFrame df
df = pd.DataFrame(data, index=data.index)[['Daily Returns', 'Volatility']]

# --- Impute NaN values using SimpleImputer ---
imputer = SimpleImputer(strategy='mean')
df[['Daily Returns', 'Volatility']] = imputer.fit_transform(df[['Daily Returns', 'Volatility']])

# --- HMM Initialization with Domain Knowledge ---
state_ranges = {
    0: {  # Bear Market
        'Daily Returns': (-0.05, -0.01),
        'Volatility': (0.005, 0.02),
    },
    1: {  # Bull Market
        'Daily Returns': (0.01, 0.05),
        'Volatility': (0.001, 0.01),
    },
    2: {  # Sideways Market
        'Daily Returns': (-0.01, 0.01),
        'Volatility': (0.0, 0.005),
    }
}

# --- Initialize the Gaussian HMM ---
np.random.seed(42)
model = hmm.GaussianHMM(
    n_components=3,
    covariance_type="full",
    n_iter=200,
    init_params="",
    min_covar=1e-5,
    params="mc"
)

# Set initial state probabilities
model.startprob_ = np.array([0.5, 0.3, 0.2])

# Set transition probabilities
model.transmat_ = np.array([
    [0.6, 0.2, 0.2],
    [0.2, 0.6, 0.2],
    [0.2, 0.2, 0.6]
])

# Initialize means based on the midpoints of the ranges
model.means_ = np.array([
    [np.mean(state_ranges[i]['Daily Returns']),
     np.mean(state_ranges[i]['Volatility'])]
    for i in range(model.n_components)
])

# Initialize covariances
model.covars_ = np.array([
    [[np.var(state_ranges[i]['Daily Returns']) + 1e-5, 0],
     [0, np.var(state_ranges[i]['Volatility']) + 1e-5]]
    for i in range(model.n_components)
])

# --- Fit the HMM model ---
model.fit(df[['Daily Returns', 'Volatility']])

# --- Predict the hidden states using Viterbi algorithm ---
hidden_states = model.predict(df[['Daily Returns', 'Volatility']])

# --- Add predicted states to DataFrame ---
df['Hidden States'] = hidden_states

# --- Group by Hidden States and calculate mean and standard deviation ---
regime_analysis = df.groupby('Hidden States').agg(['mean', 'std'])[['Daily Returns', 'Volatility']]
regime_analysis.columns = ['Avg Daily Returns', 'Std Daily Returns', 'Avg Volatility', 'Std Volatility']

# --- Function to detect regime ---
def detect_regime(observation, model):
    """
    Detects the current regime based on the given observation and HMM model.
    """
    distributions = [norm, gamma, beta, expon]
    observation = np.array(observation[:2]).flatten()
    best_fit = None
    best_fit_pvalue = 0

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        for dist in distributions:
            try:
                params = dist.fit(observation)
                _, pvalue = kstest(observation, dist.cdf, args=params)
                if pvalue > best_fit_pvalue:
                    best_fit = dist
                    best_fit_pvalue = pvalue
                    best_fit_params = params
            except Exception:
                continue

    if not best_fit:
        return 2

    regime_probabilities = []
    single_observation = observation[0]
    for state in range(model.n_components):
        pdf_value = best_fit.pdf(single_observation, *best_fit_params)
        regime_probabilities.append(pdf_value)

    regime_probabilities = np.array(regime_probabilities) / np.sum(regime_probabilities)
    prob_bear, prob_bull, prob_sideways = regime_probabilities

    if prob_bear > 0.4 and (prob_bear - prob_bull) > 0.1:
        return 0
    elif prob_bull > 0.4 and (prob_bull - prob_bear) > 0.1:
        return 1
    else:
        return 2

# --- Apply the detect_regime function ---
df['Detected Regime'] = df[['Daily Returns', 'Volatility']].apply(lambda x: detect_regime(x, model), axis=1)

# --- Debugging and Visualization ---

# Print descriptive statistics of scaled features
print("Descriptive Statistics of Scaled Features:")
print(df[['Daily Returns', 'Volatility']].describe().to_markdown(numalign="left", stralign="left"))

# Print learned model parameters
print("\nLearned Model Parameters:")
print("Means:", model.means_)
print("Covars:", model.covars_)

# Print the distribution of hidden states
print("\nDistribution of Hidden States:")
print(df['Hidden States'].value_counts().to_markdown(numalign="left", stralign="left"))

# Print the distribution of detected regimes
print("\nDistribution of Detected Regimes:")
print(df['Detected Regime'].value_counts().to_markdown(numalign="left", stralign="left"))

# Visualize the data and detected regimes
plt.figure(figsize=(12, 6))
plt.scatter(df['Daily Returns'], df['Volatility'], c=df['Detected Regime'], cmap='viridis')
plt.title('Market Regimes')
plt.xlabel('Daily Returns')
plt.ylabel('Volatility')
plt.colorbar(label='Detected Regime')
plt.show()

# --- Save the DataFrame to a CSV file ---
df.to_csv('market_regime_data.csv')


