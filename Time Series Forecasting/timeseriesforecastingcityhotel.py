# -*- coding: utf-8 -*-
"""TimeSeriesForecastingCityHotel.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16hFgGg_0H3WAhfOpvpqxHUrJjRB6BHc3
"""

import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_excel('/content/drive/My Drive/Colab Notebooks/Hotel data.xlsx')\
    .rename(columns={'arrival_date_year':'year','arrival_date_month':'month',
                     'arrival_date_day_of_month':'day'})

def monthToNum(month):
    month_dict = {
        'January': 1,
        'February': 2,
        'March': 3,
        'April': 4,
        'May': 5,
        'June': 6,
        'July': 7,
        'August': 8,
        'September': 9,
        'October': 10,
        'November': 11,
        'December': 12
    }
    return month_dict.get(month, month)  # Return the month directly if it's already a number

print(df.columns[df.columns.duplicated()])

df = df.loc[:, ~df.columns.duplicated()]

df['month'] = df['month'].apply(monthToNum)
df['date']= pd.to_datetime(df[["year", "month", "day"]])

df[['agent','company']] = df[['agent','company']].drop

df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['month'] = pd.to_numeric(df['month'], errors='coerce')
df['day'] = pd.to_numeric(df['day'], errors='coerce')

# Create the date column
df['date'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')

df.shape

df = df[(df['is_canceled']==0) & (df['hotel']=='City Hotel')]
t_df = df.groupby(['date'])['hotel'].count().reset_index()\
         .rename(columns={'hotel':'y','date':'ds'})

import plotly.express as px

fig = px.line(t_df, x="ds", y="y", title='hotel demands')
fig.show()

train_df = t_df.loc[(t_df['ds']>='2018-07-01') & (t_df['ds']<'2020-08-01')]
test_df = t_df.loc[(t_df['ds']>='2020-08-01') & (t_df['ds']<'2020-09-01')]

from pylab import rcParams
import statsmodels.api as sm

train_df_arima = train_df.copy()
train_df_arima = train_df_arima.set_index('ds')

rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(train_df_arima['y'], model='additive', period=365)  # Daily data with yearly period

fig = decomposition.plot()
plt.show()

from statsmodels.tsa.stattools import adfuller
from numpy import log

train_df_arima = train_df.copy()
train_df_arima.set_index('ds', inplace=True)
result = adfuller(train_df_arima['y'].dropna())
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

import statsmodels.api as sm
mod = sm.tsa.arima.ARIMA(train_df_arima['y'],order=(2,0,2),\
                         seasonal_order=(2,1,0,52))

results = mod.fit()

forecast_start_date = test_df['ds'].iloc[0]

# Generate the forecast (using steps=31)
forecast_results = results.forecast(steps=31)

# Create a date range for the forecast period
forecast_dates = pd.date_range(start=forecast_start_date, periods=31, freq='D')

# Create a DataFrame with the forecast and the correct dates
forecast_df = pd.DataFrame({'ds': forecast_dates, 'ARIMA_forecast': forecast_results})

# 2. Merge with Test Data (Corrected Merge):
test_output = pd.merge(test_df, forecast_df, on='ds', how='inner')


test_output['ARIMA_forecast'] = test_output['ARIMA_forecast'].astype(int)


print("\nHead of Merged Data (test_output):")
print(test_output.head())


# 4. Plotting
test_output_viz = test_output[['ds', 'y', 'ARIMA_forecast']].set_index('ds')

test_output_viz.plot.line(figsize=(7, 4), fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Bookings', fontsize=12)
plt.title('Actual vs. ARIMA Forecast')
plt.legend(['Actual Bookings', 'ARIMA Forecast'])
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

arima_mae = mean_absolute_error(test_output['y'], test_output['ARIMA_forecast'])
arima_rmse = np.sqrt(mean_squared_error(test_output['y'], test_output['ARIMA_forecast']))

print(f"ARIMA MAE: {int(arima_mae)}")
print(f"ARIMA RMSE: {int(arima_rmse)}")

from statsmodels.tsa.holtwinters import ExponentialSmoothing


seasonal_period = None  # Set to an integer if you have seasonality, otherwise None

es_model = ExponentialSmoothing(train_df['y'], trend='add', seasonal='add', seasonal_periods=10)

es_fit = es_model.fit()
es_predictions = es_fit.forecast(len(test_df))

es_forecast_df = test_df.copy() # important to copy to avoid SettingWithCopyWarning
es_forecast_df['Exponential_Smoothing_Forecast'] = es_predictions

es_forecast_df['Exponential_Smoothing_Forecast'] = es_forecast_df['Exponential_Smoothing_Forecast'].astype(int) # Convert to int if needed

es_test_output_viz = es_forecast_df[['ds', 'y', 'Exponential_Smoothing_Forecast']].set_index('ds')

# Example with additive trend and multiplicative seasonality (adjust as needed)
model = ExponentialSmoothing(train_df['y'], trend='add', seasonal='mul', seasonal_periods=10) # Example: weekly seasonality

print("Exponential Smoothing Forecasts:")
print(es_test_output_viz.head())  # Print the DataFrame with actual and predicted values

es_test_output_viz.plot.line(figsize=(7, 4), fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Bookings', fontsize=12)
plt.title('Actual vs Exponential Smoothing Forecast', fontsize=14)
plt.legend(['Actual Bookings', 'Exponential Smoothing Forecast'])
plt.show()

es_mae = mean_absolute_error(test_df['y'], es_predictions)
es_rmse = np.sqrt(mean_squared_error(test_df['y'], es_predictions))
print(f"Exponential Smoothing MAE: {int(es_mae)}")
print(f"Exponential Smoothing RMSE: {int(es_rmse)}")

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import matplotlib.pyplot as plt

train_df_prophet = train_df.copy()
model = Prophet(weekly_seasonality=True)
model.fit(train_df_prophet)
forecast = model.make_future_dataframe(periods=31)
pred = model.predict(forecast)

#plots
prophet_plot = model.plot(pred)
prophet_plot2 = model.plot_components(pred)

forecast_df = test_df.merge(pred,left_on='ds', right_on='ds', how='inner')\
.rename(columns={'yhat':'Prophet_Forecast'})

test_output_viz = forecast_df[['ds','y','Prophet_Forecast']]
test_output_viz= test_output_viz.set_index('ds')

test_output_viz.plot.line(figsize=(7,4),fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Bookings', fontsize=12)

from sklearn.metrics import mean_absolute_error , mean_squared_error
prophet_mae = mean_absolute_error(forecast_df['y'], forecast_df['Prophet_Forecast'])
prophet_rmse = np.sqrt(mean_squared_error(forecast_df['y'], forecast_df['Prophet_Forecast']))
print(f"Prophet MAE: {int(prophet_mae)}")  # Correct: Print the variable
print(f"Prophet RMSE: {int(prophet_rmse)}")