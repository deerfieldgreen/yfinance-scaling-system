import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define the stocks to analyze
stocks = ["AAPL", "META", "TSLA"]

# Create an empty DataFrame to store the overall results with correct data types
results = pd.DataFrame({
    "Stock": pd.Series(dtype="str"),
    "Mean Open": pd.Series(dtype=np.float64),
    "Std Dev Open": pd.Series(dtype=np.float64),
    "Mean Close": pd.Series(dtype=np.float64),
    "Std Dev Close": pd.Series(dtype=np.float64),
    "Mean Volume": pd.Series(dtype=np.float64),
    "Std Dev Volume": pd.Series(dtype=np.float64)
})

# Create an empty DataFrame to store the predictions
predictions_df = pd.DataFrame(columns=["Stock", "Predicted Closing Price", "Std Dev of Predictions"])

# Loop through each stock
for stock in stocks:
    # Create a Ticker object
    ticker = yf.Ticker(stock)

    # Download hourly data for the last 100 hours
    data = ticker.history(period="1mo", interval="1h")  # Fetch 5 days of data to ensure 100 hours

    # Extract data for the current stock
    stock_data = data["Close"]  # Use Close for hourly data
    open_price = data["Open"]
    volume = data["Volume"]

    # Calculate mean and standard deviation
    mean_open = float(open_price.mean())
    std_open = float(open_price.std())
    mean_close = float(stock_data.mean())
    std_close = float(stock_data.std())
    mean_volume = float(volume.mean())
    std_volume = float(volume.std())

    # Append results to the DataFrame (using concat)
    new_row = pd.DataFrame({"Stock": [stock],
                              "Mean Open": [mean_open],
                              "Std Dev Open": [std_open],
                              "Mean Close": [mean_close],
                              "Std Dev Close": [std_close],
                              "Mean Volume": [mean_volume],
                              "Std Dev Volume": [std_volume]})
    results = pd.concat([results, new_row], ignore_index=True)

   
    # Prepare data for LSTM
    # Scale the data
    scaler = MinMaxScaler()
    data["Close"] = scaler.fit_transform(np.array(data["Close"]).reshape(-1, 1))

    # Create sequences for LSTM (modified for two outputs)
    look_back = 10  # Number of previous hours to consider
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data["Close"][i:(i + look_back)])
        Y.append([data["Close"][i + look_back], np.std(data["Close"][i:(i + look_back)])]) # Close price and its std dev
    X, Y = np.array(X), np.array(Y)

    # Reshape input to be [samples, time steps, features] which is required for LSTM
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Create and train the LSTM model (modified for two outputs)
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(2))

    # Two output neurons for closing price and std dev
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs=25, batch_size=1, verbose=2)

    # Make predictions
    train_predict = model.predict(X)

    # Inverse transform the predictions to get actual closing price values
    predicted_closing_price = scaler.inverse_transform(train_predict[:, 0].reshape(-1, 1))
    actual_closing_price = scaler.inverse_transform(Y[:, 0].reshape(-1, 1))

    # Extract predicted standard deviation
    predicted_std_dev = train_predict[:, 1]

    # Append predictions to the DataFrame (corrected)
    new_pred_row = pd.DataFrame({"Stock": [stock], 
                                 "Predicted Closing Price": [predicted_closing_price[-1][0]], 
                                 "Std Dev of Predictions": [predicted_std_dev[-1]]})  # Use predicted_std_dev
    predictions_df = pd.concat([predictions_df, new_pred_row], ignore_index=True)










# Print the overall results
print("\nOverall Results:")
print(results.to_markdown(numalign="left", stralign="left"))

# Print the predictions
print("\nPredictions:")
print(predictions_df.to_markdown(numalign="left", stralign="left"))



