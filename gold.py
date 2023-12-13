import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from matplotlib import pyplot as plt

# Load gold price data
gold_data = pd.read_csv('Gold price INR.csv')  # Replace 'gold_price.csv' with the actual file name

# Display gold price data
st.title('Forecasting prices of commodities(Gold)')
st.subheader('Gold Price History')

# Plot gold price data with range slider
fig_gold = px.line(gold_data, x='Date', y='INR', title='Gold Price History', range_x=[gold_data['Date'].min(), gold_data['Date'].max()])
st.plotly_chart(fig_gold)

# Train/test split graph with different colormap
st.subheader('Train/Test Split')

train_size = st.slider('Training Data Percentage:', 0.1, 0.9, 0.7, 0.05)
train_data, test_data = train_test_split(gold_data, train_size=train_size, shuffle=False)

fig_split = go.Figure()
fig_split.add_trace(go.Scatter(x=train_data['Date'], y=train_data['INR'], mode='lines', name='Train', line=dict(color='blue')))
fig_split.add_trace(go.Scatter(x=test_data['Date'], y=test_data['INR'], mode='lines', name='Test', line=dict(color='red')))
fig_split.update_layout(title='Train/Test Split with Different Colormap', xaxis_title='Date', yaxis_title='Gold Price')
st.plotly_chart(fig_split)

# Forecasting with LSTM
st.subheader('Gold Price Forecasting with LSTM')

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Prepare data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(np.array(gold_data['INR']).reshape(-1, 1))

training_data_len = int(np.ceil(len(scaled_data) * train_size))

train_data = scaled_data[0:int(training_data_len), :]
x_train, y_train = [], []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=100, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=110, epochs=40)

# Create test dataset
test_data = scaled_data[training_data_len - 60:, :]

x_test, y_test = [], gold_data['INR'][training_data_len:].values

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Plot forecast
st.subheader('Forecast data with LSTM')
forecast_dates = pd.date_range(start=pd.to_datetime(gold_data['Date'].max()) + pd.DateOffset(1), periods=period, freq='D')
forecast_df = pd.DataFrame({'Date': forecast_dates[:len(predictions)], 'INR': predictions.flatten()[:len(forecast_dates)]})
st.write(forecast_df)

st.write(f'Forecast plot for {n_years} year/years')
fig_forecast_lstm = go.Figure()
fig_forecast_lstm.add_trace(go.Scatter(x=gold_data['Date'], y=gold_data['INR'], mode='lines', name='Actual Data'))
fig_forecast_lstm.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['INR'], mode='lines', name='Forecast', line=dict(color='green')))
fig_forecast_lstm.update_layout(title='Gold Price Forecast with LSTM', xaxis_title='Date', yaxis_title='Gold Price')
st.plotly_chart(fig_forecast_lstm)

# Regression Evaluation Metrics
st.subheader('Regression Evaluation Metrics')

# Evaluate LSTM model
mae_lstm = mean_absolute_error(y_test, predictions)
mse_lstm = mean_squared_error(y_test, predictions)
r2_lstm = r2_score(y_test, predictions)

# Display metrics for LSTM model
st.write('LSTM Model Metrics:')
st.write(f'Mean Absolute Error (MAE): {mae_lstm}')
st.write(f'Mean Squared Error (MSE): {mse_lstm}')
st.write(f'R-squared (R2): {r2_lstm}')
