import pandas as pd
import random
from scipy.stats import skew, kurtosis

# Generate a list of 100 random values between 1 and 36.77
spread_ = [random.uniform(0.978, 36.77) for _ in range(100)]

# Create a Pandas DataFrame from the list
df = pd.DataFrame({'spread_v': spread_})

# Calculate and print descriptive statistics
print("Median:", df['spread_v'].median())
print("Mode:", df['spread_v'].mode()[0])  # Mode can have multiple values, take the first one
print("Variance:", df['spread_v'].var())
print("Skewness:", skew(df['spread_v']))
print("Kurtosis:", kurtosis(df['spread_v']))

# You can also use the `describe()` method for a summary
print("\nSummary Statistics:")
print(df.describe())


