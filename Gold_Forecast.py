import streamlit as st
import pandas as pd

from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from matplotlib.dates import date2num

amazon_data = pd.read_csv('Daily_Gold_Price_on_World.csv')

START = amazon_data['Date'].min()
TODAY = amazon_data['Date'].max()

st.title('Forecasting prices of commodities(Gold)')

n_years = st.slider('Weeks of prediction:', 1, 4)
period = n_years * 7

@st.cache_data
def load_data():
    data = amazon_data.copy()
    data.reset_index(inplace=True, drop=True)
    return data

data = load_data()

st.subheader('Raw data head')
st.write(data.head())

st.subheader('Raw data tail')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['INR'], name="Gold price history"))
    #fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','INR']]
df_train = df_train.rename(columns={"Date": "ds", "INR": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data with Prophet')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} week/weeks')
fig1 = plot_plotly(m, forecast)
fig1.update_traces(line=dict(color='#0000FF'), marker=dict(color='#ADD8E6'))
st.plotly_chart(fig1)

# Prepare the data for linear regression
df_lr = df_train.copy()
df_lr['ds'] = pd.to_datetime(df_lr['ds'])  # Convert to datetime
df_lr['ds_numeric'] = (df_lr['ds'] - df_lr['ds'].min()).dt.days  # Feature: days since the start

# Prepare future data for linear regression
future_lr_dates = pd.date_range(start=df_lr['ds'].max() + pd.DateOffset(1), periods=period, freq='D')
future_lr_df = pd.DataFrame({'ds': future_lr_dates})
future_lr_values = future_lr_df['ds'].apply(lambda x: (x - df_lr['ds'].min()).days).values.reshape(-1, 1)

# Fit Linear Regression model
model = LinearRegression()
model.fit(df_lr[['ds_numeric']], df_lr['y'])

# Make predictions with Linear Regression for entire data and future data
future_lg = model.predict(df_lr[['ds_numeric']])  # Predict for entire data
predicted_future = model.predict(future_lr_values)  # Predict for future data

# Show and plot forecast from Linear Regression
st.subheader('Forecast data with Linear Regression')
st.write(pd.DataFrame({'Date': future_lr_dates, 'INR': predicted_future}))  # Display predicted values for the future

st.write(f'Forecast plot for {n_years} week/weeks')
def plot_linear_regression():
    fig = go.Figure()

    # Add scatter plot for actual data
    fig.add_trace(go.Scatter(x=df_lr['ds'], y=df_lr['y'], mode='markers', name='Actual Data'))

    # Add line plot for Linear Regression Line
    fig.add_trace(go.Scatter(x=df_lr['ds'], y=future_lg, mode='lines', name='Linear Regression Line', line=dict(color='red')))

    # Add dashed line plot for Future Predictions
    x_future = pd.date_range(start=df_lr['ds'].max() + pd.DateOffset(1), periods=len(predicted_future), freq='D')
    fig.add_trace(go.Scatter(x=x_future, y=predicted_future, mode='lines', line=dict(color='green', dash='dash'),
                             name='Future Predictions'))

    # Update layout with rangeslider and larger graphical area
    fig.update_layout(xaxis=dict(title='Days since start', rangeslider=dict(visible=True)),
                      yaxis=dict(title='INR Price'),
                      width=1000,  # Set the width of the chart
                      height=600)  # Set the height of the chart

    # Display the figure in Streamlit
    st.plotly_chart(fig)

plot_linear_regression()

# st.subheader("Forecast components")
# fig2 = m.plot_components(forecast)
# st.write(fig2)

# Evaluate Prophet model
y_true_prophet = df_train['y'].values
y_pred_prophet = forecast['yhat'].values[:-period]  # Exclude the forecasted future period

mae_prophet = mean_absolute_error(y_true_prophet, y_pred_prophet)
mse_prophet = mean_squared_error(y_true_prophet, y_pred_prophet)
r2_prophet = r2_score(y_true_prophet, y_pred_prophet)

# Evaluate Linear Regression model
y_true_lr = df_lr['y'].values
y_pred_lr = model.predict(df_lr[['ds_numeric']])

mae_lr = mean_absolute_error(y_true_lr, y_pred_lr)
mse_lr = mean_squared_error(y_true_lr, y_pred_lr)
r2_lr = r2_score(y_true_lr, y_pred_lr)

# Create a DataFrame for the metrics
metrics_data = {
    'Metric': ['Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'R-squared (R2)'],
    'Prophet Model': [mae_prophet, mse_prophet, r2_prophet],
    'Linear Regression Model': [mae_lr, mse_lr, r2_lr]
}

metrics_df = pd.DataFrame(metrics_data)

# Display the DataFrame
# st.subheader('Regression Evaluation Metrics')
# st.write(metrics_df)